"""Stable natural keys and provenance-aware lexical deduplication."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from ukstress.normalizer import canonical_stressed_form, lookup_key, stress_signature


def stable_natural_key(kind: str, *parts: object) -> str:
    payload = json.dumps(
        [kind, *parts],
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def lexeme_key(
    dataset_key: str,
    lemma_normalized: str,
    part_of_speech: str | None,
    sense_key: str,
    source_title: str,
    source_section: str,
) -> str:
    return stable_natural_key(
        "lexeme",
        dataset_key,
        lemma_normalized,
        part_of_speech,
        sense_key,
        source_title,
        source_section,
    )


@dataclass(frozen=True, order=True)
class SourceProvenance:
    page_id: int
    revision_id: int
    source_kind: str
    source_fragment: str


@dataclass(frozen=True)
class FormObservation:
    lexeme_key: str
    stressed_form: str
    morphology_key: str
    source: SourceProvenance


@dataclass(frozen=True)
class MergedForm:
    word_form_key: str
    lexeme_key: str
    form_normalized: str
    morphology_key: str
    stress_variants: tuple[tuple[str, str], ...]
    sources: tuple[SourceProvenance, ...]


def deduplicate_observations(observations: list[FormObservation]) -> list[MergedForm]:
    grouped: dict[
        tuple[str, str, str],
        tuple[set[tuple[str, str]], set[SourceProvenance]],
    ] = {}
    for observation in observations:
        normalized = lookup_key(observation.stressed_form)
        group_key = (observation.lexeme_key, normalized, observation.morphology_key)
        variants, sources = grouped.setdefault(group_key, (set(), set()))
        stressed = canonical_stressed_form(observation.stressed_form)
        variants.add((stressed, stress_signature(stressed)))
        sources.add(observation.source)

    merged: list[MergedForm] = []
    for (lexeme, normalized, morphology), (variants, sources) in sorted(grouped.items()):
        merged.append(
            MergedForm(
                word_form_key=stable_natural_key(
                    "word_form", lexeme, normalized, morphology
                ),
                lexeme_key=lexeme,
                form_normalized=normalized,
                morphology_key=morphology,
                stress_variants=tuple(sorted(variants)),
                sources=tuple(sorted(sources)),
            )
        )
    return merged
