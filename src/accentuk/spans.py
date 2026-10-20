"""Span routing for the span-to-span verbalizer.

The detector finds the parts of a sentence that may need to be spoken differently from how they are written --
digits, Latin, symbols, capitals, short abbreviations, footnote marks -- from the shape of the characters only, with
no word lists. It errs on the side of taking too much: a span that needs no change comes back unchanged from the
model, while a missed span stays written in the output. The model sees the sentence with one span marked and writes
only that span; everything outside the spans is the input, untouched.

Nothing here converts, expands or corrects text. `assemble` places the model's outputs at their spans.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

OPEN, CLOSE = "[[", "]]"

CYR = "А-Яа-яІіЇїЄєҐґ"
CYR_UP = "А-ЯІЇЄҐ"
APOS = "'’ʼ"
# Atoms, tried in order at each position. Each is a shape, never a word.
ATOMS = re.compile(
    "|".join([
        # footnote marks: [93], [*], (1), superscripts
        r"\[\s*(?:\d{1,4}|\*+)\s*\]",
        r"[¹²³⁰⁴⁵⁶⁷⁸⁹]+",
        r"[\U0001F000-\U0001FAFF☀-➿️]+",
        # Latin runs: words, e-mails, URLs, versions, with digits inside ("iPhone 17", "RTX 5090", "Wi-Fi 6E")
        r"(?:https?://|www\.)\S+",
        r"[\w.+-]*[A-Za-z][\w.+-]*@[\w-]+(?:\.[\w-]+)+",
        r"[A-Za-z][A-Za-z0-9₀-₉" + APOS + r".&+/-]*(?:[  ]+(?:[A-Za-z][A-Za-z0-9₀-₉" + APOS + r".&+/-]*|\d[\w.]*))*",
        # numbers: digit groups with inner separators, a sign or currency before, a hyphenated ending after,
        # and one short token or symbol after ("1912 р.", "2 ТБ", "3 м", "25 %", "5 °C")
        r"(?:[$€£₴№#§±~≈<>+−-]\s?)?\d(?:[\d\u00a0 .,:/×x*^°′″'-]*\d)?[°′″]?"
        r"(?:-[" + CYR + r"]+(?:[" + APOS + r"][" + CYR + r"]+)*)?"
        # then the next word too, which agrees with the number and may change with it ("571 доларів" is read
        # "п'ятсот сімдесят один долар"): a unit, a short abbreviation, or any one Cyrillic word
        r"(?:\s?(?:[%‰°℃℉$€£₴]+[CF]?|[" + CYR + r"A-Za-z]{1,3}\.?(?![" + CYR + r"A-Za-z])|[" + CYR + r"]{1,3}\.(?!\s*$)"
        r"|[" + CYR + r"]+(?:[" + APOS + r"-][" + CYR + r"]+)*))?",
        # capitals inside a word or standing alone: "ДТП", "СС", "ВІП-секторі", one capital before a period
        r"[" + CYR_UP + r"]{2,}(?:-[" + CYR + r"]+)?",
        r"(?<![" + CYR + r"])[" + CYR_UP + r"]\.",
        # two two-letter abbreviations joined by a dash with no spaces: "пн–пт", "вт—сб" (a hyphen joins words)
        r"(?<![" + CYR + APOS + r"])[а-яіїєґ]{2}[–—][а-яіїєґ]{2}(?![" + CYR + APOS + r"])",
        # a two-letter word with no vowel is an abbreviation: "пт", "сб", "нд" (Ukrainian words have a vowel)
        r"(?<![" + CYR + APOS + r"])[б-джзй-нп-тф-щьґ]{2}(?![" + CYR + APOS + r".])",
        # short lower-case abbreviations followed by more text: "т. ін.", "ст.", "обл.", "м."
        # (a chain like "т. ін." may end the sentence; a single short word before the final period is just a word)
        r"(?<![" + CYR + APOS + r"])[а-яіїєґ]{1,5}\.(?:\s?[а-яіїєґ]{1,5}\.)+",
        r"(?<![" + CYR + APOS + r"])[а-яіїєґ]{1,5}\.(?=\s*[" + CYR + r"\d,;:)»—–-])",
        # a short word before the final period may be an abbreviation ("сьогодні ср.", "на груд."); the model decides
        r"(?<![" + CYR + APOS + r"])[а-яіїєґ]{1,3}\.$",
        # stretched words: "смачнооо", "Ураааа"
        r"[" + CYR + APOS + r"]*([" + CYR + r"])\1\1[" + CYR + APOS + r"]*",
        # symbols that are read
        # hashtags and handles, a unit after a slash ("МБ/с", "км/год"), a capitalised abbreviation ("Дж.", "Див.")
        r"[#@][\w.]*\w",
        r"/[" + CYR + r"A-Za-z]{1,4}\.?(?![" + CYR + r"A-Za-z])",
        r"(?<![" + CYR + r"])[" + CYR_UP + r"][а-яіїєґ]{1,3}\.(?=\s?[" + CYR + r"\d§№])",
        r"[%‰№§&@#*+=<>/\\|~^°×÷·•→←⇒∞√∑≈≠≤≥±©®™]",
    ])
)
# a digit glued to the end of a Cyrillic word is a footnote mark ("референдарій9")
GLUED_NOTE = re.compile(r"(?<=[" + CYR + r"])\d{1,3}(?![\d" + CYR + r"])")
WORD = re.compile(r"[" + CYR + APOS + r"]+")
CLAUSE_START = re.compile(r"(?:^|[.!?…:;«\"„“(]\s*|[—–-]\s*)$")
SINGLE_CAP = re.compile(r"(?<![" + CYR + r"A-Za-z])[" + CYR_UP + r"](?![" + CYR + r"A-Za-z.])")


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    text: str


def _is_caps_run(sentence: str) -> bool:
    words = WORD.findall(sentence)
    caps = [w for w in words if len(w) > 1 and w.isupper()]
    return len(caps) >= 3 and len(caps) / max(1, len(words)) > 0.5


def detect_spans(sentence: str) -> list[Span]:
    """Spans of the sentence the model should read; ordered, non-overlapping, merged when they touch."""
    s = unicodedata.normalize("NFC", sentence)
    shouted = _is_caps_run(s)
    found: list[tuple[int, int]] = []
    for m in ATOMS.finditer(s):
        text = m.group().rstrip()
        if re.search(r"[A-Za-z]", text) and not re.fullmatch(r"[a-z]{1,4}\.", text):
            text = text.rstrip(".,;:!?)")  # a Latin run or URL does not take the sentence's closing punctuation
        if not text.strip():
            continue
        if shouted and re.fullmatch(r"[" + CYR_UP + r"]{2,}(?:-[" + CYR + r"]+)?", text):
            continue
        found.append((m.start(), m.start() + len(text)))
    for m in GLUED_NOTE.finditer(s):
        found.append((m.start(), m.end()))
    # a lone capital letter that does not open a clause: "Олександр І", "Мартіна V" (Cyrillic look-alikes)
    for m in SINGLE_CAP.finditer(s):
        if not CLAUSE_START.search(s[: m.start()]):
            found.append((m.start(), m.end()))
    # a span never cuts a word: "[[ЗАГС]]і" becomes "[[ЗАГСі]]" (a glued footnote digit stays its own span)
    glued = {(m.start(), m.end()) for m in GLUED_NOTE.finditer(s)}
    widened = []
    for a, b in found:
        if (a, b) not in glued:
            # (superscript and subscript digits are marks, not part of the word they touch)
            def word_char(c: str) -> bool:
                return c.isalpha() or c.isdecimal() or c in APOS
            while a > 0 and word_char(s[a - 1]) and (s[a].isalpha() or s[a].isdecimal()):
                a -= 1
            while b < len(s) and word_char(s[b]) and (s[b - 1].isalpha() or s[b - 1].isdecimal()):
                b += 1
        widened.append((a, b))
    found = sorted(widened)
    merged: list[list[int]] = []
    for a, b in found:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
            continue
        if merged:
            gap = s[merged[-1][1]:a]
            # ranges and runs: "3-4", "5 — 10", "067 123", "1 млн 200", "XVI ст."
            # spans with only a space, a dash, an ampersand, a bracket or a prime between them are read together:
            # "3-4", "(056) 318-00-97", "Procter & Gamble", "А. С. Макаренка", "24,7670° пн. ш."
            if re.fullmatch(r"\s?[-–—&]\s?|[ \u00a0]|\)\s?|\s?\(|\s?[′″']\s?|:", gap):
                merged[-1][1] = b
                continue
        merged.append([a, b])
    # after a number or a Roman numeral, a short abbreviation or unit that ends the clause is read with it:
    # "VIII ст.", "80-120 тис. грн", "500 тис. грн." (up to two of them)
    tail = re.compile(r"[ \u00a0]([а-яіїєґ]{1,3}\.|[а-яіїєґA-Za-z]{1,3}(?=[\s,.;:!?)»]|$))")
    for span in merged:
        if not _numeric(s[span[0]:span[1]]):
            continue
        for _ in range(2):
            m = tail.match(s, span[1])
            if not m:
                break
            span[1] = m.end()
    # the tail may now reach a span after it ("на рік" and a final "рік."): merge what overlaps
    final: list[list[int]] = []
    for a, b in merged:
        if final and a <= final[-1][1]:
            final[-1][1] = max(final[-1][1], b)
        else:
            final.append([a, b])
    return [Span(a, b, s[a:b]) for a, b in final]


def _numeric(text: str) -> bool:
    return any(ch.isdigit() for ch in text) or bool(re.fullmatch(r"[IVXLCDM]+", text))


def model_input(sentence: str, span: Span, context_words: int = 40) -> str:
    """The sentence with one span marked, cut to `context_words` words on each side."""
    s = unicodedata.normalize("NFC", sentence)
    left = s[: span.start].split(" ")
    right = s[span.end:].split(" ")
    left_text = " ".join(left[-context_words - 1:]) if len(left) > context_words + 1 else s[: span.start]
    right_text = " ".join(right[: context_words + 1]) if len(right) > context_words + 1 else s[span.end:]
    return f"{left_text}{OPEN}{span.text}{CLOSE}{right_text}"


def assemble(sentence: str, spans: list[Span], outputs: list[str]) -> str:
    """The sentence with each span replaced by the model's output for it; nothing else changes."""
    s = unicodedata.normalize("NFC", sentence)
    parts, at = [], 0
    for span, out in zip(spans, outputs, strict=True):
        before = s[at: span.start]
        if not out.strip():
            # a span spoken as nothing (a footnote mark) takes one space next to it with it
            after = s[span.end: span.end + 1]
            if before.endswith(" ") and (after in ("", " ") or after in ",.;:!?…)»"):
                before = before[:-1]
        out = out.strip() if not out.strip() else out
        # a span glued to a word ("Так/Ні", "усно/письмово") is spoken as separate words: keep them apart
        if out and before and before[-1].isalnum() and out[0].isalnum():
            out = " " + out
        after = s[span.end: span.end + 1]
        if out and after.isalnum() and out[-1].isalnum():
            out = out + " "
        parts.append(before)
        parts.append(out)
        at = span.end
    parts.append(s[at:])
    return "".join(parts).strip()
