"""Compatibility exports for the independent candidate-conditioned scorer."""

from ukstress.scorer import (
    CandidateConditionedScorer,
    CandidateScoreResult,
    CandidateTextRepresentation,
    CandidateTrainingExample,
    audio_embedding,
    compare_scorers,
)

__all__ = [
    "CandidateConditionedScorer",
    "CandidateScoreResult",
    "CandidateTextRepresentation",
    "CandidateTrainingExample",
    "audio_embedding",
    "compare_scorers",
]
