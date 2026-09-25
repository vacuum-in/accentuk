"""Heading-aware isolation of Ukrainian Wiktionary sections."""

from __future__ import annotations

import re
from dataclasses import dataclass

_HEADING = re.compile(r"^(?P<equals>={1,6})\s*(?P<title>.*?)\s*(?P=equals)\s*$", re.MULTILINE)
_UK_TEMPLATE_MARKER = re.compile(r"^\{\{=uk=.*$", re.MULTILINE)
_LANGUAGE_TEMPLATE_MARKER = re.compile(r"^\{\{=[^=\n]+=.*$", re.MULTILINE)
_UKRAINIAN_MARKERS = frozenset({"українська", "українська мова", "uk", "{{-uk-}}", "{{uk}}"})


@dataclass(frozen=True)
class Heading:
    level: int
    title: str
    start: int
    content_start: int


@dataclass(frozen=True)
class UkrainianEntry:
    heading: str
    content: str


def _headings(wikitext: str) -> list[Heading]:
    return [
        Heading(
            level=len(match.group("equals")),
            title=match.group("title").strip(),
            start=match.start(),
            content_start=match.end(),
        )
        for match in _HEADING.finditer(wikitext)
    ]


def _marker(title: str) -> str:
    return " ".join(title.casefold().split())


def ukrainian_entries(wikitext: str) -> tuple[list[UkrainianEntry], list[str]]:
    """Return only confirmed Ukrainian entries and unknown language-like markers."""
    template_entries = _template_marker_entries(wikitext)
    if template_entries:
        return template_entries, []
    headings = _headings(wikitext)
    entries: list[UkrainianEntry] = []
    unknown_markers: list[str] = []
    for index, heading in enumerate(headings):
        marker = _marker(heading.title)
        next_language = next(
            (
                candidate.start
                for candidate in headings[index + 1 :]
                if candidate.level <= heading.level
            ),
            len(wikitext),
        )
        if marker not in _UKRAINIAN_MARKERS:
            if "укра" in marker or marker.startswith("{{-"):
                unknown_markers.append(heading.title)
            continue
        nested = [
            candidate
            for candidate in headings[index + 1 :]
            if heading.content_start <= candidate.start < next_language
            and candidate.level > heading.level
        ]
        lexical = [candidate for candidate in nested if candidate.level == heading.level + 1]
        if not lexical:
            entries.append(
                UkrainianEntry(heading.title, wikitext[heading.content_start:next_language])
            )
            continue
        for entry_index, lexical_heading in enumerate(lexical):
            entry_end = (
                lexical[entry_index + 1].start
                if entry_index + 1 < len(lexical)
                else next_language
            )
            entries.append(
                UkrainianEntry(
                    lexical_heading.title,
                    wikitext[lexical_heading.content_start:entry_end],
                )
            )
    return entries, unknown_markers


def _template_marker_entries(wikitext: str) -> list[UkrainianEntry]:
    """Support current dump pages that use standalone ``{{=uk=}}`` markers."""
    entries: list[UkrainianEntry] = []
    all_markers = list(_LANGUAGE_TEMPLATE_MARKER.finditer(wikitext))
    for index, marker in enumerate(all_markers):
        if not marker.group().startswith("{{=uk="):
            continue
        end = all_markers[index + 1].start() if index + 1 < len(all_markers) else len(wikitext)
        entries.append(UkrainianEntry("Ukrainian", wikitext[marker.end() : end]))
    return entries
