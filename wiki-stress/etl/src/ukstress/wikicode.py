"""Non-executing traversal of supported MediaWiki markup."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

import mwparserfromhell

from ukstress.morphology import MorphologyCatalog
from ukstress.normalizer import validate_stress


@dataclass(frozen=True)
class ExtractedCandidate:
    stressed_form: str
    source_kind: str
    part_of_speech: str | None
    confidence: float
    grammatical_tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class UnhandledTemplateReport:
    normalized_name: str
    occurrences: int
    sample_page_titles: tuple[str, ...]
    sample_invocations: tuple[str, ...]


class UnhandledTemplateRegistry:
    def __init__(self, *, sample_limit: int = 3, fragment_limit: int = 256) -> None:
        if sample_limit < 1 or fragment_limit < 1:
            raise ValueError("sample and fragment limits must be positive")
        self._sample_limit = sample_limit
        self._fragment_limit = fragment_limit
        self._counts: Counter[str] = Counter()
        self._pages: dict[str, list[str]] = {}
        self._invocations: dict[str, list[str]] = {}

    def record(self, name: str, page_title: str, invocation: str) -> None:
        self._counts[name] += 1
        pages = self._pages.setdefault(name, [])
        samples = self._invocations.setdefault(name, [])
        bounded = invocation[: self._fragment_limit]
        if len(samples) < self._sample_limit and bounded not in samples:
            samples.append(bounded)
            pages.append(page_title)

    def reports(self) -> list[UnhandledTemplateReport]:
        return [
            UnhandledTemplateReport(
                normalized_name=name,
                occurrences=self._counts[name],
                sample_page_titles=tuple(self._pages[name]),
                sample_invocations=tuple(self._invocations[name]),
            )
            for name in sorted(self._counts)
        ]


def normalized_template_name(name: str) -> str:
    return "-".join(str(name).replace("_", " ").casefold().split())


_HEADWORD_POS = {
    "uk-noun": "noun",
    "uk-іменник": "noun",
    "uk-adj": "adjective",
    "uk-прикметник": "adjective",
    "uk-verb": "verb",
    "uk-дієслово": "verb",
    "uk-pron": "pronoun",
    "uk-num": "numeral",
    "uk-adv": "adverb",
}
_UKRAINIAN_HEADWORD_PREFIXES = (
    ("імен-uk", "noun"),
    ("прикм-uk", "adjective"),
    ("дієсл-uk", "verb"),
    ("займ-uk", "pronoun"),
    ("числ-uk", "numeral"),
    ("присл-uk", "adverb"),
)
_PRONUNCIATION_TEMPLATES = frozenset(
    {"transcription-uk", "transcriptions-uk", "транскрипція"}
)
_INFLECTION_PREFIXES = (
    "uk-decl",
    "uk-conj",
    "uk-імен",
    "uk-відм",
    "uk-діє",
    "uk-pron-decl",
    "uk-num-decl",
    "uk-part-decl",
    "uk-adv",
    "uk-cmpr",
    "uk-supr",
)
_INFLECTION_POS = {
    "uk-decl-noun": "noun",
    "uk-decl-adj": "adjective",
    "uk-conj": "verb",
    "uk-імен": "noun",
    "uk-діє": "verb",
    "uk-pron-decl": "pronoun",
    "uk-num-decl": "numeral",
    "uk-part-decl": "participle",
    "uk-cmpr": "comparative",
    "uk-supr": "superlative",
}


def _plain(value: object) -> str:
    code = mwparserfromhell.parse(str(value))
    return code.strip_code(normalize=True, collapse=True).strip()


def _candidate(
    value: object, source_kind: str, pos: str | None, confidence: float
) -> ExtractedCandidate | None:
    stressed = _plain(value)
    if not stressed or validate_stress(stressed).classification == "rejected":
        return None
    return ExtractedCandidate(stressed, source_kind, pos, confidence)


def extract_templates(
    wikitext: str,
    *,
    page_title: str = "",
    unhandled_registry: UnhandledTemplateRegistry | None = None,
    morphology_catalog: MorphologyCatalog | None = None,
) -> tuple[list[ExtractedCandidate], Counter[str]]:
    """Extract only registered template data; templates are never evaluated."""
    candidates: list[ExtractedCandidate] = []
    unhandled: Counter[str] = Counter()
    for template in mwparserfromhell.parse(wikitext).filter_templates(recursive=True):
        name = normalized_template_name(str(template.name))
        if morphology_catalog is not None:
            morphology = morphology_catalog.extract(template)
            if morphology:
                candidates.extend(
                    ExtractedCandidate(
                        form.stressed_form,
                        "structured_inflection",
                        form.part_of_speech,
                        0.95,
                        form.grammatical_tags,
                    )
                    for form in morphology
                )
                continue
        pos = _HEADWORD_POS.get(name)
        if pos is not None:
            values = [
                parameter.value
                for parameter in template.params
                if str(parameter.name).strip().casefold() in {"1", "head", "lemma", "слово"}
            ]
            for value in values:
                candidate = _candidate(value, "structured_headword", pos, 1.0)
                if candidate is not None:
                    candidates.append(candidate)
            continue
        ukrainian_pos = next(
            (
                part_of_speech
                for prefix, part_of_speech in _UKRAINIAN_HEADWORD_PREFIXES
                if name.startswith(prefix)
            ),
            None,
        )
        if ukrainian_pos is not None:
            for parameter in template.params:
                candidate = _candidate(
                    parameter.value, "structured_headword", ukrainian_pos, 1.0
                )
                if candidate is not None:
                    candidates.append(candidate)
            continue
        if name in _PRONUNCIATION_TEMPLATES:
            for parameter in template.params:
                candidate = _candidate(parameter.value, "pronunciation", None, 0.9)
                if candidate is not None:
                    candidates.append(candidate)
            continue
        if name.startswith(_INFLECTION_PREFIXES):
            inflection_pos = next(
                (
                    family_pos
                    for prefix, family_pos in _INFLECTION_POS.items()
                    if name.startswith(prefix)
                ),
                None,
            )
            for parameter in template.params:
                candidate = _candidate(
                    parameter.value, "structured_inflection", inflection_pos, 0.95
                )
                if candidate is not None:
                    candidates.append(candidate)
            continue
        unhandled[name] += 1
        if unhandled_registry is not None:
            unhandled_registry.record(name, page_title, str(template))
    return candidates, unhandled


def extract_bold_headwords(wikitext: str) -> list[ExtractedCandidate]:
    """Controlled fallback for bold forms in an already-confirmed Ukrainian entry."""
    candidates: list[ExtractedCandidate] = []
    for match in re.finditer(r"'''(?P<word>[^']+)'''", wikitext):
        candidate = _candidate(match.group("word"), "bold_headword", None, 0.9)
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def extract_controlled_fallback(
    wikitext: str, *, max_candidates: int = 50
) -> list[ExtractedCandidate]:
    """Extract explicit stressed tokens from a confirmed Ukrainian section as a last resort."""
    if max_candidates < 1:
        raise ValueError("max_candidates must be positive")
    plain = _plain(wikitext)
    candidates: list[ExtractedCandidate] = []
    seen: set[str] = set()
    for token in re.split(r"[\s,;:()]+", plain):
        value = token.strip(".!?«»\"")
        if "\u0301" not in value or value in seen:
            continue
        candidate = _candidate(value, "controlled_fallback", None, 0.5)
        if candidate is not None:
            candidates.append(candidate)
            seen.add(value)
        if len(candidates) >= max_candidates:
            break
    return candidates
