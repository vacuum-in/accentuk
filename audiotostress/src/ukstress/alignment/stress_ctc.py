"""Optional adapter for stress-aware Ukrainian CTC/ASR checkpoints."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ukstress.datasets import StressCandidate
from ukstress.lexicon.stress import canonicalize_stressed_form, normalize_surface


@dataclass(frozen=True)
class StressCTCOutput:
    stressed_text: str
    confidence: float | None = None
    model_id: str | None = None


@dataclass(frozen=True)
class StressCTCVote:
    candidate_index: int | None
    confidence: float | None
    model_id: str | None
    reason: str | None = None


class StressCTCAdapter:
    """Normalize an external stress-aware model into a lexicon candidate vote."""

    def __init__(
        self,
        runner: Callable[[np.ndarray, int, str], StressCTCOutput | str | dict[str, Any]],
        *,
        model_id: str,
    ) -> None:
        if not model_id.strip():
            raise ValueError("model_id must be non-empty")
        self.runner = runner
        self.model_id = model_id

    def predict(
        self,
        waveform: np.ndarray,
        sample_rate: int,
        target_word: str,
        candidates: Sequence[StressCandidate],
    ) -> StressCTCVote:
        if sample_rate <= 0 or waveform.ndim != 1:
            raise ValueError(
                "CTC input must be a one-dimensional waveform and positive sample rate"
            )
        if not candidates:
            raise ValueError("at least one lexicon candidate is required")
        output = self._normalize_output(self.runner(waveform, sample_rate, target_word))
        try:
            stressed = canonicalize_stressed_form(output.stressed_text)
        except ValueError:
            return StressCTCVote(
                None, output.confidence, output.model_id or self.model_id, "invalid_stress_output"
            )
        if normalize_surface(stressed) != normalize_surface(target_word):
            return StressCTCVote(
                None,
                output.confidence,
                output.model_id or self.model_id,
                "target_identity_mismatch",
            )
        matches = [
            index
            for index, candidate in enumerate(candidates)
            if canonicalize_stressed_form(candidate.stressed_form) == stressed
        ]
        if len(matches) != 1:
            reason = "candidate_mapping_ambiguous" if len(matches) > 1 else "candidate_not_found"
            return StressCTCVote(None, output.confidence, output.model_id or self.model_id, reason)
        return StressCTCVote(matches[0], output.confidence, output.model_id or self.model_id)

    def _normalize_output(self, raw: StressCTCOutput | str | dict[str, Any]) -> StressCTCOutput:
        if isinstance(raw, StressCTCOutput):
            return raw
        if isinstance(raw, str):
            return StressCTCOutput(raw, model_id=self.model_id)
        if isinstance(raw, dict):
            text = next(
                (
                    raw.get(key)
                    for key in ("stressed_text", "stressed_form", "text", "transcript")
                    if isinstance(raw.get(key), str)
                ),
                None,
            )
            if text is None:
                raise ValueError(
                    "stress CTC runner must return stressed_text or a normalized output"
                )
            confidence = raw.get("confidence")
            return StressCTCOutput(
                text,
                float(confidence) if confidence is not None else None,
                str(raw.get("model_id", self.model_id)),
            )
        raise ValueError("stress CTC runner must return stressed_text or a normalized output")
