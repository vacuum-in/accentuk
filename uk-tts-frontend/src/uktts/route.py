"""Decide whether a chunk needs the verbalizer at all.

This is routing, not verbalization: it inspects the text and picks an owner —
the model, or pass-through unchanged. It never rewrites a character, and the
verbalizer's own policy names this as permitted runtime logic.

The reason it exists is measured. Sent every sentence unconditionally, the
checkpoint alters **209 of 880** (23.8%) plain Ukrainian sentences from lang-uk's
benchmark — sentences with no digit, no Latin letter and nothing else to spell
out. The failures are not small:

    сидів спокійно       ->  сидівох
    на березі            ->  від трьох до нуля
    незрозумілих явищах  ->  трьох цілих дев'яти десятих відсотка

A sequence-to-sequence model asked to copy its input will sometimes decline to.
Text with nothing to verbalize has nothing to gain from the attempt, so it does
not make it.
"""

from __future__ import annotations

import re
import unicodedata

from .segment import ABBREVIATIONS

# Anything a voice cannot read off the page as Ukrainian words.
DIGIT = re.compile(r"\d", re.UNICODE)
LATIN = re.compile(r"[A-Za-z]")
# Currency, units, operators and reference marks. Sentence punctuation, the
# hyphen, quotes and the apostrophe are ordinary text and are not listed.
SYMBOL = re.compile(r"[%$€£¥₴°№§©®™+=×÷<>≤≥≠~^@&*/\\|_\[\]{}]")
# `ООН`, `США`, `КНР` — read letter by letter, not as a word. Bounded at five
# letters: a heading is set in capitals too, and `ІСТОРИЧНИЙ НАРИС` sent to the
# model came back `і ес те оРИЧНИЙ НАРИС`.
CAPITALS = re.compile(
    r"(?<![^\W\d_])[A-ZА-ЯІЇЄҐ]{2,5}(?![^\W\d_])"
)
WORD = re.compile(r"[^\W\d_]+(?:['’ʼ-][^\W\d_]+)*", re.UNICODE)
# Superscript and subscript digits are digits a `\d` class does not match.
SCRIPT_DIGITS = frozenset("⁰¹²³⁴⁵⁶⁷⁸⁹₀₁₂₃₄₅₆₇₈₉")


def needs_verbalization(text: str) -> bool:
    """True when the chunk contains something that must be spelled out.

    Errs towards the model: anything unrecognised is sent, because a missed
    number is a number read as silence, while a needless pass-through only
    costs the improvement the model might have made — and on plain text it
    makes none.
    """

    value = unicodedata.normalize("NFC", text)
    if DIGIT.search(value) or LATIN.search(value) or SYMBOL.search(value):
        return True
    if SCRIPT_DIGITS & set(value):
        return True
    # A line that is entirely capitals is a heading, not a run of acronyms.
    letters = [c for c in value if c.isalpha()]
    if letters and not all(c.isupper() for c in letters) and CAPITALS.search(value):
        return True
    # `2024 р.` is caught by the digit; `кілька тис. осіб` is not, and the
    # abbreviation still has to be read out in full.
    for match in WORD.finditer(value):
        end = match.end()
        word = match.group()
        # A single letter before a period is an initial — `С. Смеречинський` —
        # and nothing can expand it. Sent to the model, it was dropped outright.
        if len(word) < 2:
            continue
        if end < len(value) and value[end] == "." and word.lower() in ABBREVIATIONS:
            return True
    return False
