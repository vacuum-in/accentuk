"""Normative Unicode normalization for Ukrainian stress-lexicon records."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

ACUTE = "\u0301"
BREVE = "\u0306"
DIAERESIS = "\u0308"
CANONICAL_APOSTROPHE = "\u02bc"
APOSTROPHE_VARIANTS = "'`\u2018\u2019\u02bb\uff07"
HYPHEN_VARIANTS = "\u2010\u2011\u2012\u2013\u2014\u2212"
UKRAINIAN_LETTERS = frozenset("абвгґдеєжзиіїйклмнопрстуфхцчшщьюя")
UKRAINIAN_VOWELS = frozenset("аеєиіїоуюя")
_INTERNAL_HYPHEN = re.compile(f"(?<=\\S)[{HYPHEN_VARIANTS}](?=\\S)")
_TOKEN_BOUNDARY = re.compile(r"[\s-]+")


class NormalizationError(ValueError):
    """Raised when a value cannot be represented as a supported lexical form."""


@dataclass(frozen=True)
class ValidationResult:
    """Classification used by extraction before database publication."""

    classification: str
    reason: str | None = None


def canonical_stressed_form(value: str) -> str:
    """Return NFD text with canonical apostrophes and internal hyphens."""
    normalized = unicodedata.normalize("NFD", value)
    apostrophe_map = str.maketrans({char: CANONICAL_APOSTROPHE for char in APOSTROPHE_VARIANTS})
    normalized = normalized.translate(apostrophe_map)
    return _INTERNAL_HYPHEN.sub("-", normalized)


def lookup_key(value: str) -> str:
    """Return the lowercase key, removing only the configured acute accent."""
    return canonical_stressed_form(value).lower().replace(ACUTE, "")


def validate_characters(value: str) -> ValidationResult:
    """Reject controls, unsupported scripts, and combining marks other than acute."""
    if not value:
        return ValidationResult("rejected", "empty form")
    canonical = canonical_stressed_form(value)
    for index, character in enumerate(canonical):
        if character in {CANONICAL_APOSTROPHE, "-", " ", "\t", ACUTE}:
            continue
        if character == BREVE and index > 0 and canonical[index - 1].lower() == "и":
            continue
        if character == DIAERESIS and index > 0 and canonical[index - 1].lower() == "і":
            continue
        if character.lower() in UKRAINIAN_LETTERS:
            continue
        if unicodedata.category(character).startswith("C"):
            return ValidationResult("rejected", "control character")
        if unicodedata.combining(character):
            return ValidationResult("rejected", "unsupported combining mark")
        return ValidationResult("rejected", f"unsupported character: {character!r}")
    return ValidationResult("valid")


def _stressed_letter(canonical: str, acute_index: int) -> str | None:
    """Return the composed letter an acute attaches to, or None at a boundary.

    In NFD text `ї` is `і` plus a combining diaeresis and `й` is `и` plus a
    combining breve, so the code point immediately before the acute is not the
    letter being stressed. The base and its non-acute marks are recomposed, which
    keeps stressed `ї` valid while still rejecting stressed `й`.
    """
    position = acute_index - 1
    while position >= 0 and canonical[position] in {BREVE, DIAERESIS}:
        position -= 1
    if position < 0:
        return None
    return unicodedata.normalize("NFC", canonical[position:acute_index])


def validate_stress(value: str) -> ValidationResult:
    """Validate that every acute immediately follows a Ukrainian vowel."""
    canonical = canonical_stressed_form(value)
    character_result = validate_characters(canonical)
    if character_result.classification == "rejected":
        return character_result
    for index, character in enumerate(canonical):
        if character != ACUTE:
            continue
        letter = _stressed_letter(canonical, index)
        if letter is None or letter.lower() not in UKRAINIAN_VOWELS:
            return ValidationResult("rejected", "acute must immediately follow a Ukrainian vowel")
    return ValidationResult("valid")


def stress_signature(value: str) -> str:
    """Return vowel-ordinal stress positions, never byte or code-point offsets."""
    result = validate_stress(value)
    if result.classification == "rejected":
        raise NormalizationError(result.reason or "invalid stressed form")

    canonical = canonical_stressed_form(value)
    segments = [segment for segment in _TOKEN_BOUNDARY.split(canonical) if segment]
    signatures: list[tuple[int, int]] = []
    for segment_index, segment in enumerate(segments):
        vowel_ordinal = -1
        for character in segment:
            if character.lower() in UKRAINIAN_VOWELS:
                vowel_ordinal += 1
            if character == ACUTE:
                signatures.append((segment_index, vowel_ordinal))

    if not signatures:
        raise NormalizationError("stressed form has no acute accent")
    if len(segments) == 1:
        return "|".join(str(vowel_ordinal) for _, vowel_ordinal in signatures)
    return "|".join(
        f"{segment_index}:{vowel_ordinal}" for segment_index, vowel_ordinal in signatures
    )
