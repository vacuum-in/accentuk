"""Independent candidate-conditioned audio/text scoring."""

from ukstress.scorer.core import (
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
