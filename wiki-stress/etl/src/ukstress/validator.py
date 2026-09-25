"""Candidate validation and quarantine classification."""

from __future__ import annotations

import re
from dataclasses import dataclass

from ukstress.normalizer import ACUTE, canonical_stressed_form, validate_stress

_RESIDUAL_MARKUP = re.compile(r"<[^>]*>|\{\{|\}\}|\[\[|\]\]")


@dataclass(frozen=True)
class CandidateValidation:
    classification: str
    canonical_form: str
    is_multiword: bool
    reasons: tuple[str, ...] = ()


def classify_candidate(value: str) -> CandidateValidation:
    """Classify a candidate as valid, plausible, or rejected with stable reasons."""
    canonical = canonical_stressed_form(value.strip())
    is_multiword = bool(re.search(r"\s", canonical))
    if _RESIDUAL_MARKUP.search(canonical):
        return CandidateValidation(
            "rejected", canonical, is_multiword, ("residual HTML or Wikicode",)
        )
    stress_result = validate_stress(canonical)
    if stress_result.classification == "rejected":
        return CandidateValidation(
            "rejected",
            canonical,
            is_multiword,
            (stress_result.reason or "invalid characters or stress",),
        )
    if ACUTE not in canonical:
        return CandidateValidation(
            "plausible", canonical, is_multiword, ("no explicit stress mark",)
        )
    return CandidateValidation("valid", canonical, is_multiword)

