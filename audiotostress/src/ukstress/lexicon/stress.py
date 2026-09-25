"""Canonical Ukrainian stress marks and vowel indices."""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass

UKRAINIAN_VOWELS = frozenset("аеєиіїоуюя")
COMBINING_ACUTE = "\N{COMBINING ACUTE ACCENT}"
_POSTFIX_STRESS_MARKS = frozenset(
    {
        COMBINING_ACUTE,
        "\N{ACUTE ACCENT}",
        "\N{MODIFIER LETTER ACUTE ACCENT}",
    }
)
_APOSTROPHES = str.maketrans({"’": "'", "‘": "'", "ʼ": "'", "`": "'"})
_HYPHENS = str.maketrans({"‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-", "−": "-"})


@dataclass(frozen=True)
class ParsedStress:
    surface: str
    stressed_form: str
    vowel_index: int


def _normalize_punctuation(text: str) -> str:
    return text.translate(_APOSTROPHES).translate(_HYPHENS)


def normalize_surface(text: str) -> str:
    """Normalize case, Unicode, apostrophes/hyphens, and remove stress marks."""

    if not isinstance(text, str) or not text.strip() or "\0" in text:
        raise ValueError("surface form must be non-empty text without NUL")
    value = unicodedata.normalize("NFC", _normalize_punctuation(text.strip()).casefold())
    value = "".join(character for character in value if character not in _POSTFIX_STRESS_MARKS)
    value = value.replace("+", "")
    if any(character.isspace() for character in value):
        raise ValueError("surface form must contain exactly one token")
    return unicodedata.normalize("NFC", value)


def vowel_count(surface: str) -> int:
    return sum(character in UKRAINIAN_VOWELS for character in normalize_surface(surface))


def apply_stress(surface: str, vowel_index: int) -> str:
    """Place a canonical combining acute after a zero-based vowel index."""

    plain = normalize_surface(surface)
    if isinstance(vowel_index, bool) or vowel_index < 0:
        raise ValueError("vowel_index must be a non-negative integer")
    seen = 0
    output: list[str] = []
    placed = False
    for character in plain:
        output.append(character)
        if character in UKRAINIAN_VOWELS:
            if seen == vowel_index:
                output.append(COMBINING_ACUTE)
                placed = True
            seen += 1
    if not placed:
        raise ValueError(f"vowel_index {vowel_index} is outside a {seen}-vowel surface")
    return unicodedata.normalize("NFC", "".join(output))


def parse_stress(stressed_form: str) -> ParsedStress:
    """Parse common acute/plus notation into one canonical stressed form."""

    if not isinstance(stressed_form, str) or not stressed_form.strip() or "\0" in stressed_form:
        raise ValueError("stressed form must be non-empty text without NUL")
    value = unicodedata.normalize("NFC", _normalize_punctuation(stressed_form.strip()).casefold())
    marks: list[tuple[int, int]] = []
    vowel_position = -1
    characters = list(value)
    for position, character in enumerate(characters):
        if character in UKRAINIAN_VOWELS:
            vowel_position += 1
        if character in _POSTFIX_STRESS_MARKS:
            if position == 0 or characters[position - 1] not in UKRAINIAN_VOWELS:
                raise ValueError("stress mark must immediately follow a Ukrainian vowel")
            marks.append((position, vowel_position))
        elif character == "+":
            if position + 1 >= len(characters) or characters[position + 1] not in UKRAINIAN_VOWELS:
                raise ValueError("plus stress mark must immediately precede a Ukrainian vowel")
            marks.append((position, vowel_position + 1))
    if len(marks) != 1:
        raise ValueError("stressed form must contain exactly one stress mark")
    stress_index = marks[0][1]
    surface = normalize_surface(value)
    return ParsedStress(
        surface=surface,
        stressed_form=apply_stress(surface, stress_index),
        vowel_index=stress_index,
    )


def canonicalize_stressed_form(stressed_form: str) -> str:
    return parse_stress(stressed_form).stressed_form
