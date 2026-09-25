"""Manual-gold calibrated ensemble confidence and production gating."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np

from ukstress.calibration import precision_coverage_curve, validate_precision_gate
from ukstress.provenance.fingerprints import file_fingerprint


@dataclass(frozen=True)
class EnsembleCalibrator:
    """A monotonic confidence calibrator fitted only on held-out manual gold."""

    temperature: float
    version: str = "ensemble-temperature-v1"

    @classmethod
    def fit(
        cls,
        raw_confidence: np.ndarray,
        correct: np.ndarray,
        *,
        split_name: str,
        manual_gold: bool,
    ) -> EnsembleCalibrator:
        if split_name != "validation" or not manual_gold:
            raise ValueError("ensemble calibration requires held-out manual validation data")
        if (
            raw_confidence.ndim != 1
            or correct.shape != raw_confidence.shape
            or not len(raw_confidence)
        ):
            raise ValueError("confidence and correctness must be non-empty matching vectors")
        if np.any((raw_confidence < 0.0) | (raw_confidence > 1.0)):
            raise ValueError("confidence must be in [0, 1]")
        logits = np.log(
            np.clip(raw_confidence, 1e-6, 1 - 1e-6) / np.clip(1 - raw_confidence, 1e-6, 1.0)
        )
        temperatures = np.exp(np.linspace(np.log(0.05), np.log(10.0), 100))
        losses = []
        for temperature in temperatures:
            scaled = logits / temperature
            probability = 1.0 / (1.0 + np.exp(-scaled))
            losses.append(
                float(
                    -np.mean(
                        correct * np.log(probability + 1e-12)
                        + (1 - correct) * np.log(1 - probability + 1e-12)
                    )
                )
            )
        return cls(float(temperatures[int(np.argmin(losses))]))

    def transform(self, raw_confidence: np.ndarray) -> np.ndarray:
        logits = np.log(
            np.clip(raw_confidence, 1e-6, 1 - 1e-6) / np.clip(1 - raw_confidence, 1e-6, 1.0)
        )
        return cast(np.ndarray, 1.0 / (1.0 + np.exp(-logits / self.temperature)))


@dataclass(frozen=True)
class HighPrecisionProfile:
    threshold: float
    target_precision: float
    validated: bool
    accepted: int
    precision: float
    coverage: float
    precision_lower_bound: float
    curve: list[dict[str, float | int]]

    def write(self, path: str | Path) -> str:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "profile": "high-precision",
            "threshold": self.threshold,
            "target_precision": self.target_precision,
            "validated": self.validated,
            "accepted": self.accepted,
            "precision": self.precision,
            "coverage": self.coverage,
            "precision_lower_bound": self.precision_lower_bound,
            "curve": self.curve,
        }
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        try:
            temporary.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        return file_fingerprint(destination, namespace="high-precision-profile")


def select_high_precision_profile(
    confidence: np.ndarray,
    correct: np.ndarray,
    *,
    target_precision: float = 0.99,
    min_samples: int = 30,
) -> HighPrecisionProfile:
    """Choose the highest-coverage threshold meeting the configured manual-gold gate."""

    curve = precision_coverage_curve(confidence, correct)
    eligible = [
        point
        for point in curve
        if int(point["accepted"]) >= min_samples and float(point["precision"]) >= target_precision
    ]
    point = max(eligible, key=lambda item: float(item["coverage"])) if eligible else curve[0]
    gate = validate_precision_gate(
        confidence,
        correct,
        threshold=float(point["threshold"]),
        target_precision=target_precision,
        min_samples=min_samples,
    )
    return HighPrecisionProfile(
        threshold=float(point["threshold"]),
        target_precision=target_precision,
        validated=bool(gate["validated"]),
        accepted=int(gate["accepted"]),
        precision=float(gate["precision"]),
        coverage=float(gate["coverage"]),
        precision_lower_bound=float(gate["precision_lower_bound"]),
        curve=curve,
    )
