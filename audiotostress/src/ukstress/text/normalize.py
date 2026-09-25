"""Conservative transcript normalization for token matching."""

from __future__ import annotations

import re
import unicodedata

from ukstress.lexicon.stress import COMBINING_ACUTE

NORMALIZATION_VERSION = "uk-transcript-v1"
_WHITESPACE = re.compile(r"\s+")
_STANDALONE_HYPHEN = re.compile(r"(?<!\w)-|-(?!\w)")
_APOSTROPHES = str.maketrans({"’": "'", "‘": "'", "ʼ": "'", "`": "'"})
_HYPHENS = str.maketrans({"‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-", "−": "-"})
_STRESS_MARKS = {COMBINING_ACUTE, "\N{ACUTE ACCENT}", "\N{MODIFIER LETTER ACUTE ACCENT}"}


def normalize_transcript(text: str) -> str:
    """Normalize matching text without selecting or inserting lexical stress."""

    if not isinstance(text, str) or "\0" in text:
        raise ValueError("transcript must be text without NUL")
    value = text.casefold().translate(_APOSTROPHES).translate(_HYPHENS)
    value = unicodedata.normalize("NFC", value)
    value = _STANDALONE_HYPHEN.sub(" ", value)
    output: list[str] = []
    for character in value:
        if character in _STRESS_MARKS or character == "+":
            continue
        category = unicodedata.category(character)
        if character.isalnum() or character in {"'", "-"}:
            output.append(character)
        elif category.startswith(("P", "S", "Z", "C")) or character.isspace():
            output.append(" ")
        else:
            output.append(character)
    return _WHITESPACE.sub(" ", "".join(output)).strip()
