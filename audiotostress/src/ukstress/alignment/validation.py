"""Alignment quality normalization and topology/identity checks."""

from __future__ import annotations

import statistics

from pydantic import BaseModel, ConfigDict, Field

from ukstress.alignment.interface import WordAlignmentResult
from ukstress.asr.crosscheck import transcript_identity_score
from ukstress.datasets import WordAlignment
from ukstress.text import normalize_transcript


class AlignmentValidation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    valid: bool
    quality: float = Field(ge=0.0, le=1.0)
    identity_score: float = Field(ge=0.0, le=1.0)
    mean_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    reasons: list[str]


def alignment_quality(
    words: list[WordAlignment], expected_transcript: str
) -> tuple[float, float, float | None]:
    """Combine token identity and available CTC confidence into [0, 1]."""

    aligned_text = " ".join(word.normalized_token for word in words)
    identity = transcript_identity_score(expected_transcript, aligned_text)
    confidences = [word.confidence for word in words if word.confidence is not None]
    mean_confidence = statistics.fmean(confidences) if confidences else None
    quality = identity if mean_confidence is None else 0.6 * identity + 0.4 * mean_confidence
    return max(0.0, min(1.0, quality)), identity, mean_confidence


def validate_word_alignments(
    result: WordAlignmentResult,
    expected_transcript: str,
    *,
    min_quality: float = 0.8,
    overlap_tolerance_s: float = 0.005,
) -> AlignmentValidation:
    reasons: list[str] = []
    if not result.words:
        reasons.append("word_alignment_failed")
    previous_end = 0.0
    for word in result.words:
        if word.end_s > result.utterance_duration_s + 1e-6:
            reasons.append("alignment_out_of_bounds")
        if word.start_s < previous_end - overlap_tolerance_s:
            reasons.append("alignment_overlap")
        previous_end = max(previous_end, word.end_s)
        if word.backend != result.backend or word.model_id != result.model_id:
            reasons.append("alignment_backend_mismatch")
        if word.char_start is not None and word.char_end is not None:
            matched = expected_transcript[word.char_start : word.char_end]
            if normalize_transcript(matched) != word.normalized_token:
                reasons.append("target_identity_mismatch")

    quality, identity_score, mean_confidence = alignment_quality(result.words, expected_transcript)
    if identity_score < 1.0:
        reasons.append("target_identity_mismatch")
    if quality < min_quality:
        reasons.append("low_alignment_quality")
    unique_reasons = list(dict.fromkeys(reasons))
    return AlignmentValidation(
        valid=not unique_reasons,
        quality=quality,
        identity_score=identity_score,
        mean_confidence=mean_confidence,
        reasons=unique_reasons,
    )


def find_target_alignment(
    result: WordAlignmentResult, target: str, *, occurrence: int = 0
) -> WordAlignment | None:
    if occurrence < 0:
        raise ValueError("target occurrence must be non-negative")
    normalized_target = normalize_transcript(target)
    matches = [word for word in result.words if word.normalized_token == normalized_target]
    return matches[occurrence] if occurrence < len(matches) else None
