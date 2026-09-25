"""A small, dependency-light candidate-conditioned compatibility scorer.

The scorer deliberately has a separate representation path from the vowel ranker: audio is
reduced to robust waveform statistics and text is encoded from its stressed spelling.  A
production run can replace the projection with a trained model without changing the interface.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np

from ukstress.datasets import StressCandidate


@dataclass(frozen=True)
class CandidateScoreResult:
    scores: np.ndarray
    probabilities: np.ndarray
    predicted_index: int


@dataclass(frozen=True)
class CandidateTrainingExample:
    """A scorer training item with explicit provenance for its gold decision."""

    waveform: np.ndarray
    candidates: list[str]
    gold_index: int
    source_type: str = "trusted_unambiguous"


class CandidateTextRepresentation:
    """Deterministic character/stress-aware candidate representation."""

    dimension = 16

    def encode(self, stressed_form: str) -> np.ndarray:
        if not isinstance(stressed_form, str) or not stressed_form.strip():
            raise ValueError("candidate text must be non-empty")
        digest = hashlib.sha256(stressed_form.casefold().encode("utf-8")).digest()
        values = np.frombuffer(digest[: self.dimension], dtype=np.uint8).astype(np.float32)
        values = values / 127.5 - 1.0
        return values

    def encode_candidate(self, candidate: StressCandidate) -> np.ndarray:
        """Encode a lexicon candidate while retaining its explicit stress spelling."""

        return self.encode(candidate.stressed_form)


def audio_embedding(waveform: np.ndarray, *, dimension: int = 16) -> np.ndarray:
    """Summarize a target crop into a normalized, fixed-size audio embedding."""

    if waveform.ndim != 1 or waveform.size < 2 or dimension < 4:
        raise ValueError("waveform must be a one-dimensional crop and dimension must be >= 4")
    signal = np.asarray(waveform, dtype=np.float32)
    chunks = np.array_split(signal, dimension - 3)
    means = np.asarray([float(np.mean(chunk)) for chunk in chunks], dtype=np.float32)
    global_stats = np.asarray(
        [
            float(np.sqrt(np.mean(signal * signal))),
            float(np.max(np.abs(signal))),
            float(np.mean(np.abs(np.diff(signal)))),
        ],
        dtype=np.float32,
    )
    vector = np.concatenate((means, global_stats))
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-8 else vector


class CandidateConditionedScorer:
    """Score each stressed candidate against one target audio crop."""

    def __init__(self, *, text_representation: CandidateTextRepresentation | None = None) -> None:
        self.text_representation = text_representation or CandidateTextRepresentation()
        self.candidate_bias = np.zeros(0, dtype=np.float32)

    def score(self, waveform: np.ndarray, candidates: list[str]) -> CandidateScoreResult:
        if not candidates:
            raise ValueError("at least one candidate is required")
        audio = audio_embedding(waveform, dimension=self.text_representation.dimension)
        text = np.asarray([self.text_representation.encode(candidate) for candidate in candidates])
        text_norm = np.linalg.norm(text, axis=1, keepdims=True)
        normalized = text / np.maximum(text_norm, 1e-8)
        if len(self.candidate_bias) == len(candidates):
            scores = normalized @ audio + self.candidate_bias
        else:
            scores = normalized @ audio
        probabilities = _softmax(scores)
        return CandidateScoreResult(
            scores=scores.astype(np.float32),
            probabilities=probabilities.astype(np.float32),
            predicted_index=int(np.argmax(scores)),
        )

    def fit(
        self,
        examples: Sequence[CandidateTrainingExample | tuple[np.ndarray, list[str], int]],
        *,
        epochs: int = 5,
        learning_rate: float = 0.1,
    ) -> None:
        """Fit a compact ranking bias on trusted or manually verified examples.

        The audio/text encoders remain independent from the vowel ranker; this update only
        calibrates candidate compatibility and is intentionally suitable for a small baseline.
        """

        if not examples or epochs < 1 or learning_rate <= 0.0:
            raise ValueError("fit requires examples, positive epochs, and learning_rate")
        normalized: list[CandidateTrainingExample] = []
        for example in examples:
            item = (
                example
                if isinstance(example, CandidateTrainingExample)
                else CandidateTrainingExample(*example)
            )
            if item.source_type not in {"trusted_unambiguous", "manual_ambiguous"}:
                raise ValueError("ambiguous pseudo-labels are not valid scorer gold")
            normalized.append(item)
        candidate_count = max(len(item.candidates) for item in normalized)
        self.candidate_bias = np.zeros(candidate_count, dtype=np.float32)
        for _ in range(epochs):
            for item in normalized:
                if not 0 <= item.gold_index < len(item.candidates):
                    raise ValueError("gold candidate index is outside candidates")
                result = self.score(item.waveform, item.candidates)
                predicted = result.predicted_index
                if predicted != item.gold_index:
                    self.candidate_bias[item.gold_index] += learning_rate
                    self.candidate_bias[predicted] -= learning_rate


def compare_scorers(
    ranker_predictions: Sequence[int],
    scorer_predictions: Sequence[int],
    gold: Sequence[int],
) -> dict[str, float | int]:
    """Report agreement, accuracy, and overlapping errors for ensemble diagnostics."""

    if not (len(ranker_predictions) == len(scorer_predictions) == len(gold)) or not gold:
        raise ValueError("prediction vectors must be non-empty and have matching lengths")
    ranker = np.asarray(ranker_predictions)
    scorer = np.asarray(scorer_predictions)
    truth = np.asarray(gold)
    ranker_errors = ranker != truth
    scorer_errors = scorer != truth
    return {
        "records": len(gold),
        "agreement": float(np.mean(ranker == scorer)),
        "ranker_accuracy": float(np.mean(~ranker_errors)),
        "scorer_accuracy": float(np.mean(~scorer_errors)),
        "error_overlap": int(np.sum(ranker_errors & scorer_errors)),
        "ranker_errors": int(np.sum(ranker_errors)),
        "scorer_errors": int(np.sum(scorer_errors)),
    }


def _softmax(values: np.ndarray) -> np.ndarray:
    centered = values - np.max(values)
    weights = np.exp(centered)
    return cast(np.ndarray, weights / np.sum(weights))
