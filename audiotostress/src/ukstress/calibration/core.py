"""Temperature scaling and optional PAV isotonic calibration."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np


@dataclass(frozen=True)
class TemperatureCalibrator:
    temperature: float

    @classmethod
    def fit(
        cls,
        logits: np.ndarray,
        targets: np.ndarray,
        *,
        split_name: str,
        manual_gold: bool,
    ) -> TemperatureCalibrator:
        _require_held_out_manual(split_name, manual_gold)
        if logits.ndim != 2 or targets.shape != (logits.shape[0],):
            raise ValueError("logits must be [records, classes] with matching targets")
        candidates = np.exp(np.linspace(math.log(0.05), math.log(10.0), 200))
        losses = [_cross_entropy(logits / temperature, targets) for temperature in candidates]
        return cls(float(candidates[int(np.argmin(losses))]))

    def transform(self, logits: np.ndarray) -> np.ndarray:
        return _softmax(logits / self.temperature)


@dataclass(frozen=True)
class IsotonicCalibrator:
    thresholds: tuple[float, ...]
    values: tuple[float, ...]

    @classmethod
    def fit(
        cls,
        confidence: np.ndarray,
        correct: np.ndarray,
        *,
        split_name: str,
        manual_gold: bool,
    ) -> IsotonicCalibrator:
        _require_held_out_manual(split_name, manual_gold)
        if confidence.ndim != 1 or correct.shape != confidence.shape:
            raise ValueError("confidence and correctness must be matching vectors")
        order = np.argsort(confidence)
        blocks: list[tuple[float, float, int]] = []
        for index in order:
            blocks.append((float(confidence[index]), float(correct[index]), 1))
            while len(blocks) > 1 and blocks[-2][1] / blocks[-2][2] > blocks[-1][1] / blocks[-1][2]:
                left, right = blocks.pop(-2), blocks.pop()
                blocks.append((right[0], left[1] + right[1], left[2] + right[2]))
        thresholds = tuple(block[0] for block in blocks)
        values = tuple(block[1] / block[2] for block in blocks)
        return cls(thresholds, values)

    def transform(self, confidence: np.ndarray) -> np.ndarray:
        indices = np.searchsorted(np.asarray(self.thresholds), confidence, side="left")
        indices = np.clip(indices, 0, len(self.values) - 1)
        return cast(np.ndarray, np.asarray(self.values, dtype=np.float64)[indices])


def reliability_data(
    confidence: np.ndarray,
    correct: np.ndarray,
    *,
    bins: int = 10,
) -> list[dict[str, float | int]]:
    """Return deterministic equal-width reliability-bin observations."""

    if bins < 1 or confidence.ndim != 1 or correct.shape != confidence.shape:
        raise ValueError("confidence and correctness must be matching vectors")
    if np.any((confidence < 0.0) | (confidence > 1.0)):
        raise ValueError("confidence must be in [0, 1]")
    result: list[dict[str, float | int]] = []
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        selected = (confidence >= lower) & (
            confidence < upper if index < bins - 1 else confidence <= upper
        )
        count = int(np.sum(selected))
        result.append(
            {
                "bin": index,
                "lower": lower,
                "upper": upper,
                "count": count,
                "mean_confidence": float(np.mean(confidence[selected])) if count else 0.0,
                "accuracy": float(np.mean(correct[selected])) if count else 0.0,
            }
        )
    return result


def precision_coverage_curve(
    confidence: np.ndarray,
    correct: np.ndarray,
    *,
    thresholds: np.ndarray | None = None,
) -> list[dict[str, float | int]]:
    """Measure accepted precision and coverage at descending confidence cutoffs."""

    if confidence.ndim != 1 or correct.shape != confidence.shape or not len(confidence):
        raise ValueError("confidence and correctness must be a non-empty matching vector")
    if np.any((confidence < 0.0) | (confidence > 1.0)):
        raise ValueError("confidence must be in [0, 1]")
    cutoffs = np.unique(
        np.asarray(thresholds, dtype=np.float64)
        if thresholds is not None
        else np.linspace(0.0, 1.0, 21)
    )
    if np.any((cutoffs < 0.0) | (cutoffs > 1.0)):
        raise ValueError("thresholds must be in [0, 1]")
    result: list[dict[str, float | int]] = []
    for threshold in sorted((float(value) for value in cutoffs), reverse=True):
        selected = confidence >= threshold
        count = int(np.sum(selected))
        result.append(
            {
                "threshold": threshold,
                "accepted": count,
                "precision": float(np.mean(correct[selected])) if count else 0.0,
                "coverage": count / len(confidence),
            }
        )
    return result


@dataclass(frozen=True)
class CalibrationArtifact:
    """Persisted calibration metadata and reliability/coverage evidence."""

    method: str
    version: str
    split_name: str
    manual_gold: bool
    parameters: dict[str, object]
    reliability: list[dict[str, float | int]]
    precision_coverage: list[dict[str, float | int]]

    def write(self, path: str | Path) -> str:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "method": self.method,
            "version": self.version,
            "split_name": self.split_name,
            "manual_gold": self.manual_gold,
            "parameters": self.parameters,
            "reliability": self.reliability,
            "precision_coverage": self.precision_coverage,
        }
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        try:
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8"
            )
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        import hashlib

        digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        return f"sha256:{digest}"


def validate_precision_gate(
    confidence: np.ndarray,
    correct: np.ndarray,
    *,
    threshold: float,
    target_precision: float = 0.99,
    min_samples: int = 30,
) -> dict[str, float | int | bool | str]:
    """Validate an accepted-set precision target with a Wilson lower bound."""

    if not 0.0 <= threshold <= 1.0 or not 0.0 < target_precision <= 1.0:
        raise ValueError("threshold and target_precision must be in valid probability ranges")
    curve = precision_coverage_curve(confidence, correct, thresholds=np.asarray([threshold]))
    point = curve[0]
    accepted = int(point["accepted"])
    successes = int(np.sum(correct[confidence >= threshold]))
    lower = _wilson_lower_bound(successes, accepted)
    validated = accepted >= min_samples and lower >= target_precision
    return {
        "validated": validated,
        "reason": "validated" if validated else "insufficient_precision_or_sample_count",
        "threshold": threshold,
        "target_precision": target_precision,
        "accepted": accepted,
        "precision": float(point["precision"]),
        "coverage": float(point["coverage"]),
        "precision_lower_bound": lower,
        "min_samples": min_samples,
    }


def _wilson_lower_bound(successes: int, trials: int, *, z: float = 1.96) -> float:
    if trials < 1:
        return 0.0
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = proportion + z * z / (2.0 * trials)
    spread = z * math.sqrt(proportion * (1.0 - proportion) / trials + z * z / (4.0 * trials**2))
    return max(0.0, (center - spread) / denominator)


def _require_held_out_manual(split_name: str, manual_gold: bool) -> None:
    if split_name != "validation" or not manual_gold:
        raise ValueError("calibration may fit only held-out manually verified validation data")


def _softmax(logits: np.ndarray) -> np.ndarray:
    centered = logits - np.max(logits, axis=1, keepdims=True)
    weights = np.exp(centered)
    probabilities = weights / np.sum(weights, axis=1, keepdims=True)
    return cast(np.ndarray, probabilities)


def _cross_entropy(logits: np.ndarray, targets: np.ndarray) -> float:
    probabilities = _softmax(logits)
    return float(-np.mean(np.log(probabilities[np.arange(len(targets)), targets] + 1e-12)))
