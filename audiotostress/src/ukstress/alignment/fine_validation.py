"""Validation for canonical vowel intervals produced by fine aligners."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from ukstress.datasets import RejectionRecord, VowelInterval
from ukstress.lexicon.stress import vowel_count


class VowelAlignmentValidation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    valid: bool
    expected_count: int = Field(ge=0)
    observed_count: int = Field(ge=0)
    reasons: list[str]


def fine_alignment_rejection(
    *,
    source_id: str,
    utterance_id: str,
    record_id: str | None,
    validation: VowelAlignmentValidation,
    provenance: dict[str, str] | None = None,
) -> RejectionRecord | None:
    """Convert unusable fine alignment diagnostics into a persisted rejection record."""

    if validation.valid:
        return None
    return RejectionRecord(
        source_id=source_id,
        utterance_id=utterance_id,
        record_id=record_id,
        stage="fine_alignment",
        reasons=list(validation.reasons),
        provenance=dict(provenance or {}),
    )


def validate_vowel_intervals(
    vowels: list[VowelInterval],
    normalized_word: str,
    *,
    word_start_s: float,
    word_end_s: float,
    utterance_duration_s: float | None = None,
    overlap_tolerance_s: float = 0.005,
) -> VowelAlignmentValidation:
    """Check count, ordering, overlap, and containment before feature extraction."""

    reasons: list[str] = []
    expected_count = vowel_count(normalized_word)
    if word_start_s < 0.0 or word_start_s >= word_end_s:
        reasons.append("word_interval_invalid")
    if utterance_duration_s is not None and word_end_s > utterance_duration_s + 1e-6:
        reasons.append("word_interval_out_of_bounds")
    if len(vowels) != expected_count:
        reasons.append("vowel_count_mismatch")

    previous_end = word_start_s
    for interval in vowels:
        if interval.start_s < word_start_s - overlap_tolerance_s:
            reasons.append("vowel_outside_word")
        if interval.end_s > word_end_s + overlap_tolerance_s:
            reasons.append("vowel_outside_word")
        if utterance_duration_s is not None and interval.end_s > utterance_duration_s + 1e-6:
            reasons.append("vowel_out_of_bounds")
        if interval.start_s < previous_end - overlap_tolerance_s:
            reasons.append("vowel_overlap")
        previous_end = max(previous_end, interval.end_s)

    unique_reasons = list(dict.fromkeys(reasons))
    return VowelAlignmentValidation(
        valid=not unique_reasons,
        expected_count=expected_count,
        observed_count=len(vowels),
        reasons=unique_reasons,
    )
