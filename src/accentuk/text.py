"""Orthographic tokenization, signature rendering, and capitalization filtering."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

from .normalizer import ACUTE, UKRAINIAN_VOWELS, lookup_key

JOINERS = "-'ʼ’‘ʻ`＇‐‑‒–—−"
BREAKS = "-‐‑‒–—−"


@dataclass
class StressToken:
    start: int
    end: int
    text: str
    output_text: str
    status: str = ""
    candidates: list[str] = field(default_factory=list)
    scores: dict[str, float] | None = None
    margin: float | None = None


def is_letter(c: str) -> bool:
    c = c.lower()
    return "а" <= c <= "я" or c in "єіїґ"


def word_tokens(text: str) -> list[StressToken]:
    result: list[StressToken] = []
    start: int | None = None
    last: int | None = None
    for i, c in enumerate(text):
        if is_letter(c) or (start is not None and unicodedata.category(c) == "Mn"):
            if start is None:
                start = i
            last = i + 1
        elif start is not None and c in JOINERS:
            continue
        elif start is not None:
            raw = text[start:last]
            result.append(StressToken(start, last, raw, raw))
            start = last = None
    if start is not None:
        raw = text[start:last]
        result.append(StressToken(start, last, raw, raw))
    return result


def counts_as_vowel(c: str) -> bool:
    """Return whether a character counts as a vowel in Ukrainian stress signatures."""
    if not c:
        return False
    decomposed = unicodedata.normalize("NFD", c.lower())
    return decomposed[0] in UKRAINIAN_VOWELS


def vowel_count(text: str) -> int:
    return sum(c in UKRAINIAN_VOWELS for c in lookup_key(text))


def apply_signature(surface: str, signature: str) -> str:
    """Apply a stress signature (e.g. '0', '1', '1:0', '0|1') to a surface word."""
    if ACUTE in surface or not signature:
        raise ValueError("invalid surface or empty signature")
    if "|" in signature and ":" not in signature:
        signature = signature.split("|")[0]
    wanted: set[tuple[int, int]] = set()
    for part in signature.split("|"):
        if not re.fullmatch(r"(?:\d+:)?\d+", part):
            raise ValueError("invalid signature")
        pair = tuple(map(int, part.split(":"))) if ":" in part else (0, int(part))
        wanted.add(pair)
    out: list[str] = []
    segment, ordinal, inserted, i = 0, -1, 0, 0
    while i < len(surface):
        c = surface[i]
        if c in BREAKS or c.isspace():
            segment += 1
            ordinal = -1
        out.append(c)
        i += 1
        if not counts_as_vowel(c):
            continue
        ordinal += 1
        while i < len(surface) and unicodedata.category(surface[i]) == "Mn":
            out.append(surface[i])
            i += 1
        if (segment, ordinal) in wanted:
            out.append(ACUTE)
            inserted += 1
    if inserted != len(wanted):
        raise ValueError("signature does not fit surface")
    return "".join(out)


def sentence_initial(text: str, start: int) -> bool:
    """Check if token at start opens a sentence."""
    for c in reversed(text[:start]):
        if c.isspace() or c in "«»\"'„“”‘’()[]—–-":
            continue
        return c in ".!?…:"
    return True


def signature_of_output(output: str) -> str:
    """Return the vowel ordinal the acute follows in a stressed surface; '' if none."""
    ordinal = -1
    for r in unicodedata.normalize("NFD", output):
        if r == ACUTE:
            if ordinal < 0:
                return ""
            return str(ordinal)
        if unicodedata.category(r) != "Mn" and counts_as_vowel(r):
            ordinal += 1
    return ""


def by_capitalisation(
    candidates: list[str],
    proper: dict[str, bool] | set[str] | Any,
    text: str,
    initial: bool,
    form: str | None = None,
) -> list[str]:
    """Narrow or reorder candidates based on capitalization."""
    if len(candidates) < 2 or not text:
        return candidates[:]

    def _is_proper(c: str) -> bool:
        if isinstance(proper, (set, frozenset)):
            return c in proper or (form is not None and (form, c) in proper)
        if isinstance(proper, dict):
            if form is not None:
                key = f"{form}\x00{c}"
                if key in proper:
                    return bool(proper[key])
                if (form, c) in proper:
                    return bool(proper[(form, c)])
            if c in proper:
                return bool(proper[c])
        return False

    names = [c for c in candidates if _is_proper(c)]
    common = [c for c in candidates if not _is_proper(c)]
    if not names or not common:
        return candidates[:]

    preferred: list[str] = common if not text[0].isupper() else names if not initial else []
    if not preferred:
        return candidates[:]
    if len(preferred) == 1:
        return preferred
    return preferred + [c for c in candidates if c not in preferred]
