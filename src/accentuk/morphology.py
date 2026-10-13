"""Tier 2: morphological resolution of grammatical stress ambiguities."""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .triage import (
    RELIABLE_SEPARATING_FEATURES,
    has_unreliable_pos_split,
    separating_features,
)

Reading = tuple[list[str], list[int]]


@dataclass(frozen=True)
class Resolution:
    accents: tuple[int, ...]
    matched_readings: int
    reason: str


_EQUIVALENT_UPOS = {
    "PRON": ("PRON", "DET"),
    "DET": ("DET", "PRON"),
    "NOUN": ("NOUN", "PROPN"),
    "PROPN": ("PROPN", "NOUN"),
}

COUNTED_NUMERALS = frozenset({"два", "дві", "три", "чотири", "обидва", "обидві"})
_COUNTED_NUMERAL_CASES = frozenset({"Nom", "Acc"})
_AGREEING_BETWEEN = frozenset({"ADJ", "DET", "NUM"})

PHASE_VERBS = frozenset({
    "почати", "почав", "почала", "почало", "почали", "починати", "починає",
    "починають", "починав", "починала", "починали",
    "стати", "став", "стала", "стало", "стали",
    "продовжити", "продовжив", "продовжила", "продовжили", "продовжувати",
    "продовжував", "продовжувала", "продовжували", "продовжує", "продовжують",
    "перестати", "перестав", "перестала", "перестали", "переставати",
    "припинити", "припинив", "припинила", "припинили", "припиняти",
    "закінчити", "закінчив", "закінчила", "закінчили",
    "взятися", "взявся", "взялася", "взялися", "братися", "береться",
})


def _feats_to_dict(feats: str | None) -> dict[str, str]:
    return dict(part.split("=", 1) for part in (feats or "").split("|") if "=" in part)


def counted_form_feats(upos: str | None, feats: str | None) -> str | None:
    """Rewrite a nominative plural as a genitive singular, or return None."""
    if upos not in {"NOUN", "PROPN"}:
        return None
    parts = _feats_to_dict(feats)
    if parts.get("Case") not in ("Nom", "Acc") or parts.get("Number") != "Plur":
        return None
    parts["Case"], parts["Number"] = "Gen", "Sing"
    return "|".join(f"{key}={value}" for key, value in sorted(parts.items()))


def governed_by_counted_numeral(preceding: list[tuple[str, str, str]]) -> bool:
    """Does a 2/3/4 numeral govern the token these tokens precede?"""
    for upos, feats, text in reversed(preceding):
        if upos not in _AGREEING_BETWEEN:
            return False
        if upos != "NUM":
            return False
        if text.lower() not in COUNTED_NUMERALS:
            return False
        case = _feats_to_dict(feats).get("Case")
        return case is None or case in _COUNTED_NUMERAL_CASES
    return False


def apply_counted_form(
    spans: dict[tuple[int, int], tuple[str, str]],
    tokens: list[tuple[tuple[int, int], str, str, str]],
    known_forms: frozenset[str] | None = None,
    enabled: bool = True,
) -> set[tuple[int, int]]:
    """Rewrite in place the parses a counted-form numeral governs."""
    rewritten_spans: set[tuple[int, int]] = set()
    if not enabled:
        return rewritten_spans
    for index, (span, upos, feats, text) in enumerate(tokens):
        if known_forms and unicodedata.normalize("NFC", text).lower() not in known_forms:
            continue
        if not governed_by_counted_numeral([(u, f, x) for _, u, f, x in tokens[:index]]):
            continue
        rewritten = counted_form_feats(upos, feats)
        if rewritten is not None:
            spans[span] = (upos, rewritten)
            rewritten_spans.add(span)
    return rewritten_spans


def later_accent(accents: tuple[int, ...]) -> int:
    """Rightmost accent position, the one that orders an aspect pair."""
    return max(accents) if accents else -1


def imperfective_infinitive(
    form: str,
    readings: list[Reading],
    preceding: list[tuple[str, str, str]],
) -> tuple[int, ...] | None:
    """The imperfective reading, when a phase verb governs this infinitive."""
    infinitives = [
        (tuple(accents), frozenset(tags))
        for tags, accents in readings
        if "VerbForm=Inf" in tags and "upos=VERB" in tags
    ]
    if len(infinitives) < 2:
        return None
    if len({tags for _, tags in infinitives}) != 1:
        return None
    ordered = sorted({a for a, _ in infinitives}, key=later_accent)
    if len(ordered) != 2:
        return None
    imperfective = [ordered[-1]]
    if later_accent(ordered[0]) == later_accent(ordered[-1]):
        return None
    for _, _, text in reversed(preceding):
        lowered = unicodedata.normalize("NFC", text).lower()
        if lowered in PHASE_VERBS:
            return imperfective[0]
        if lowered in ("не", "б", "би", "ж", "же"):
            continue
        return None
    return None


def _feature_set(upos: str | None, feats: str | None) -> set[str]:
    """Flatten a parse into the tag vocabulary the trie's readings use."""
    observed = set((feats or "").split("|")) - {""}
    for label in _EQUIVALENT_UPOS.get(upos or "", (upos,)):
        if label:
            observed.add(f"upos={label}")
    return observed


def resolve(readings: list[Reading], upos: str | None, feats: str | None) -> Resolution | None:
    """Pick the accent whose tags the parse satisfies, or None."""
    if len(readings) < 2:
        return None
    if has_unreliable_pos_split(readings):
        return None
    features = separating_features(readings)
    if not (features & RELIABLE_SEPARATING_FEATURES):
        return None

    observed = _feature_set(upos, feats)
    matched = [tuple(accents) for tags, accents in readings if set(tags) <= observed]
    if not matched:
        return None
    distinct = set(matched)
    if len(distinct) > 1:
        return None
    return Resolution(next(iter(distinct)), len(matched), "tags matched")


def apply_accents(surface: str, accents: tuple[int, ...], acute: str = "\u0301") -> str:
    """Insert the acute at character offsets."""
    chars = list(surface)
    for pos in sorted(accents, reverse=True):
        chars.insert(pos, acute)
    return "".join(chars)


def agreement_repair(
    readings: list[Reading],
    sentence: str,
    spans: dict[tuple[int, int], tuple],
    start: int,
    morph: Any = None,
) -> tuple[int, ...] | None:
    """The accent the modifier before the target demands, when the tagger misread it."""
    from .combiner import modifier_agreement

    if morph is None:
        try:
            import pymorphy3

            morph = pymorphy3.MorphAnalyzer(lang="uk")
        except Exception:
            return None

    ordered = sorted(spans)
    position = next((k for k, (a, _) in enumerate(ordered) if a == start), None)
    if not position:
        return None
    a, b = ordered[position - 1]
    if spans[(a, b)][0] not in ("DET", "ADJ"):
        return None
    if not readings or not all("upos=NOUN" in tags for tags, _ in readings):
        return None
    allowed = modifier_agreement(sentence[a:b].lower(), morph)
    if not allowed:
        return None
    tagged = _feats_to_dict(spans[(a, b)][1])
    if "Case" not in tagged or (tagged.get("Case"), tagged.get("Number")) in allowed:
        return None
    matching = set()
    for tags, accents in readings:
        own = _feats_to_dict("|".join(tags))
        if (own.get("Case"), own.get("Number")) in allowed:
            matching.add(tuple(accents))
    return next(iter(matching)) if len(matching) == 1 else None


class SpacyMorphologyTier:
    """spaCy-backed morphology resolver."""

    def __init__(
        self,
        model_path: Path | str,
        *,
        counted_forms: frozenset[str] | None = None,
        counted_form_enabled: bool = True,
        nlp: Any = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.counted_forms = counted_forms
        self.counted_form_enabled = counted_form_enabled
        self._nlp = nlp

    @property
    def pipeline(self) -> Any:
        if self._nlp is None:
            import spacy

            self._nlp = spacy.load(
                str(self.model_path), exclude=["ner", "lemmatizer", "parser", "senter"]
            )
        return self._nlp

    def parse(self, sentence: str) -> dict[tuple[int, int], tuple[str, str]]:
        return self.parse_batch([sentence])[0]

    def parse_batch(self, sentences: list[str]) -> list[dict[tuple[int, int], tuple[str, str]]]:
        if not sentences:
            return []
        results: list[dict[tuple[int, int], tuple[str, str]]] = []
        for doc in self.pipeline.pipe(sentences):
            spans: dict[tuple[int, int], tuple[str, str]] = {}
            tokens: list[tuple[tuple[int, int], str, str, str]] = []
            for token in doc:
                span = (token.idx, token.idx + len(token.text))
                spans[span] = (token.pos_, str(token.morph))
                tokens.append((span, token.pos_, str(token.morph), token.text))
            for span in apply_counted_form(
                spans, tokens, self.counted_forms, self.counted_form_enabled
            ):
                spans[span] = (*spans[span], True)
            results.append(spans)
        return results
