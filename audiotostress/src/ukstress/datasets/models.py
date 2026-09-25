"""Canonical, backend-neutral pipeline records."""

from __future__ import annotations

import math
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA_VERSION = "1.0"
SchemaVersion = Literal["1.0"]


class CanonicalModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


def _validate_identifier(value: str) -> str:
    if not value.strip() or "\0" in value:
        raise ValueError("identifier must be non-empty and cannot contain NUL")
    return value


class CorpusManifestRecord(CanonicalModel):
    schema_version: SchemaVersion = "1.0"
    source_id: str
    utterance_id: str
    audio_uri: str
    transcript: str | None = None
    speaker_id: str | None = None
    segment_start_s: float | None = Field(default=None, ge=0.0)
    segment_end_s: float | None = Field(default=None, gt=0.0)
    language: str = "uk"
    license_id: str | None = None
    commercial_use: bool | None = None
    redistribution: bool | None = None
    provenance: dict[str, str] = Field(default_factory=dict)

    _identifiers = field_validator("source_id", "utterance_id")(_validate_identifier)

    @field_validator("audio_uri")
    @classmethod
    def audio_uri_is_safe_text(cls, value: str) -> str:
        if not value.strip() or "\0" in value:
            raise ValueError("audio_uri must be non-empty and cannot contain NUL")
        return value

    @field_validator("language")
    @classmethod
    def language_is_ukrainian(cls, value: str) -> str:
        if value.casefold().split("-")[0] != "uk":
            raise ValueError("language must be Ukrainian (uk or uk-*)")
        return value

    @model_validator(mode="after")
    def segment_is_complete_and_ordered(self) -> Self:
        if (self.segment_start_s is None) != (self.segment_end_s is None):
            raise ValueError("segment_start_s and segment_end_s must be specified together")
        if (
            self.segment_start_s is not None
            and self.segment_end_s is not None
            and self.segment_start_s >= self.segment_end_s
        ):
            raise ValueError("segment start must be before segment end")
        return self


class NormalizedUtterance(CanonicalModel):
    schema_version: SchemaVersion = "1.0"
    record_id: str
    source_id: str
    utterance_id: str
    audio_uri: str
    transcript_original: str
    transcript_normalized: str
    speaker_id: str | None = None
    sample_rate: int = Field(gt=0)
    duration_s: float = Field(gt=0.0)
    source_sample_rate: int = Field(gt=0)
    source_channels: int = Field(gt=0)
    segment_start_s: float = Field(default=0.0, ge=0.0)
    transformations: list[str] = Field(default_factory=list)
    license_id: str | None = None
    commercial_use: bool | None = None
    redistribution: bool | None = None
    provenance: dict[str, str] = Field(default_factory=dict)

    _identifiers = field_validator("record_id", "source_id", "utterance_id")(_validate_identifier)


class WordAlignment(CanonicalModel):
    schema_version: SchemaVersion = "1.0"
    token: str
    normalized_token: str
    start_s: float = Field(ge=0.0)
    end_s: float = Field(gt=0.0)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    char_start: int | None = Field(default=None, ge=0)
    char_end: int | None = Field(default=None, gt=0)
    backend: str
    model_id: str | None = None

    @model_validator(mode="after")
    def bounds_are_ordered(self) -> Self:
        if self.start_s >= self.end_s:
            raise ValueError("word alignment start must be before end")
        if (self.char_start is None) != (self.char_end is None):
            raise ValueError("char_start and char_end must be specified together")
        if (
            self.char_start is not None
            and self.char_end is not None
            and self.char_start >= self.char_end
        ):
            raise ValueError("character start must be before end")
        return self


class VowelInterval(CanonicalModel):
    schema_version: SchemaVersion = "1.0"
    vowel_index: int = Field(ge=0)
    grapheme: str = Field(min_length=1, max_length=2)
    phone: str | None = None
    start_s: float = Field(ge=0.0)
    end_s: float = Field(gt=0.0)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def bounds_are_ordered(self) -> Self:
        if self.start_s >= self.end_s:
            raise ValueError("vowel interval start must be before end")
        return self


class StressCandidate(CanonicalModel):
    stressed_form: str = Field(min_length=1)
    vowel_index: int = Field(ge=0)
    source: str = Field(min_length=1)


class FeatureExample(CanonicalModel):
    schema_version: SchemaVersion = "1.0"
    record_id: str
    source_id: str
    utterance_id: str
    speaker_id: str | None = None
    audio_uri: str
    target_word: str
    word_start_s: float = Field(ge=0.0)
    word_end_s: float = Field(gt=0.0)
    vowels: list[VowelInterval] = Field(min_length=1)
    gold_vowel_index: int = Field(ge=0)
    ssl_features: list[list[float]] | None = None
    prosodic_features: list[dict[str, float | bool | None]] = Field(default_factory=list)
    lexicon_fingerprint: str
    config_fingerprint: str
    license_id: str | None = None
    provenance: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def feature_shapes_match_vowels(self) -> Self:
        if self.word_start_s >= self.word_end_s:
            raise ValueError("word start must be before end")
        indices = {vowel.vowel_index for vowel in self.vowels}
        if self.gold_vowel_index not in indices:
            raise ValueError("gold_vowel_index must identify one aligned vowel")
        if self.ssl_features is not None and len(self.ssl_features) != len(self.vowels):
            raise ValueError("ssl_features must contain one vector per vowel")
        if self.prosodic_features and len(self.prosodic_features) != len(self.vowels):
            raise ValueError("prosodic_features must contain one mapping per vowel")
        return self


class RejectionRecord(CanonicalModel):
    schema_version: SchemaVersion = "1.0"
    source_id: str
    utterance_id: str
    record_id: str | None = None
    stage: str
    reasons: list[str] = Field(min_length=1)
    detail: str | None = None
    provenance: dict[str, str] = Field(default_factory=dict)


class MiningRecord(CanonicalModel):
    schema_version: SchemaVersion = "1.0"
    record_id: str
    source_id: str
    utterance_id: str
    speaker_id: str | None = None
    audio_uri: str
    sentence_original: str
    sentence_normalized: str
    target_word: str
    target_char_start: int | None = Field(default=None, ge=0)
    target_char_end: int | None = Field(default=None, gt=0)
    word_start_s: float = Field(ge=0.0)
    word_end_s: float = Field(gt=0.0)
    vowels: list[VowelInterval] = Field(min_length=1)
    candidates: list[StressCandidate] = Field(min_length=1)
    ranker_probs: list[float] | None = None
    candidate_scorer_probs: list[float] | None = None
    stress_ctc_candidate: int | None = Field(default=None, ge=0)
    predicted_candidate: int | None = Field(default=None, ge=0)
    calibrated_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    candidate_margin: float | None = Field(default=None, ge=0.0, le=1.0)
    alignment_score: float | None = Field(default=None, ge=0.0, le=1.0)
    identity_score: float | None = Field(default=None, ge=0.0, le=1.0)
    accepted: bool
    rejection_reasons: list[str] = Field(default_factory=list)
    lexicon_fingerprint: str
    model_versions: dict[str, str] = Field(default_factory=dict)
    config_fingerprint: str
    license_id: str | None = None
    commercial_use: bool | None = None
    redistribution: bool | None = None
    provenance: dict[str, str] = Field(default_factory=dict)

    @field_validator("ranker_probs", "candidate_scorer_probs")
    @classmethod
    def probabilities_are_valid(cls, value: list[float] | None) -> list[float] | None:
        invalid = value is not None and any(
            not math.isfinite(item) or not 0 <= item <= 1 for item in value
        )
        if invalid:
            raise ValueError("candidate probabilities must be finite and in [0, 1]")
        return value

    @model_validator(mode="after")
    def decision_is_consistent(self) -> Self:
        if self.word_start_s >= self.word_end_s:
            raise ValueError("word start must be before end")
        if (self.target_char_start is None) != (self.target_char_end is None):
            raise ValueError("target character bounds must be specified together")
        if (
            self.target_char_start is not None
            and self.target_char_end is not None
            and self.target_char_start >= self.target_char_end
        ):
            raise ValueError("target character start must be before end")
        candidate_count = len(self.candidates)
        for name, probabilities in (
            ("ranker_probs", self.ranker_probs),
            ("candidate_scorer_probs", self.candidate_scorer_probs),
        ):
            if probabilities is not None and len(probabilities) != candidate_count:
                raise ValueError(f"{name} must contain one value per candidate")
        for name, candidate in (
            ("predicted_candidate", self.predicted_candidate),
            ("stress_ctc_candidate", self.stress_ctc_candidate),
        ):
            if candidate is not None and candidate >= candidate_count:
                raise ValueError(f"{name} is outside the candidate list")
        if self.accepted and self.rejection_reasons:
            raise ValueError("accepted records cannot have rejection reasons")
        if self.accepted and self.predicted_candidate is None:
            raise ValueError("accepted records require a predicted candidate")
        if not self.accepted and not self.rejection_reasons:
            raise ValueError("rejected records require at least one rejection reason")
        return self
