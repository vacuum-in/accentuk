"""Decide whether a chunk needs the verbalizer at all."""

from __future__ import annotations

import re
import unicodedata

from .segment import ABBREVIATIONS

DIGIT = re.compile(r"\d", re.UNICODE)
LATIN = re.compile(r"[A-Za-z]")
SYMBOL = re.compile(r"[%$€£¥₴°№§©®™+=×÷<>≤≥≠~^@&*/\\|_\[\]{}]")
CAPITALS = re.compile(r"(?<![^\W\d_])[A-ZА-ЯІЇЄҐ]{2,5}(?![^\W\d_])")
WORD = re.compile(r"[^\W\d_]+(?:['’ʼ-][^\W\d_]+)*", re.UNICODE)
SCRIPT_DIGITS = frozenset("⁰¹²³⁴⁵⁶⁷⁸⁹₀₁₂₃₄₅₆₇₈₉")


def needs_verbalization(text: str) -> bool:
    """Return True when the chunk contains something that must be spelled out."""
    value = unicodedata.normalize("NFC", text)
    if DIGIT.search(value) or LATIN.search(value) or SYMBOL.search(value):
        return True
    if SCRIPT_DIGITS & set(value):
        return True
    letters = [c for c in value if c.isalpha()]
    if letters and not all(c.isupper() for c in letters) and CAPITALS.search(value):
        return True
    for match in WORD.finditer(value):
        end = match.end()
        word = match.group()
        if len(word) < 2:
            continue
        if end < len(value) and value[end] == "." and word.lower() in ABBREVIATIONS:
            return True
    return False
