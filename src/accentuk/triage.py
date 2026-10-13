"""Triage utilities for classifying Ukrainian stress ambiguities."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

RELIABLE_SEPARATING_FEATURES = frozenset({"Case", "Number", "Gender", "Animacy"})
UNRELIABLE_POS_SPLIT = frozenset({"NOUN", "VERB"})

Reading = tuple[list[str], list[int]]


def _feature_map(tags: Iterable[str]) -> dict[str, str]:
    return {tag.split("=", 1)[0]: tag.split("=", 1)[1] for tag in tags if "=" in tag}


def separating_features(readings: Iterable[Reading]) -> set[str]:
    """Return the tag keys whose values differ across accent groups."""
    materialized = [(list(tags), tuple(accents)) for tags, accents in readings]
    tags_by_accent: dict[tuple[int, ...], list[list[str]]] = defaultdict(list)
    for tags, accents in materialized:
        tags_by_accent[accents].append(tags)
    if len(tags_by_accent) < 2:
        return set()
    keys = set().union(*(set(_feature_map(tags)) for tags, _ in materialized))
    groups = sorted(tags_by_accent)
    distinguishing = set()
    for key in keys:
        observed = [
            frozenset(_feature_map(tags).get(key) for tags in tags_by_accent[accents])
            for accents in groups
        ]
        if len(set(observed)) > 1:
            distinguishing.add(key)
    return distinguishing


def has_unreliable_pos_split(readings: Iterable[Reading]) -> bool:
    """True when a noun reading and a verb reading carry different accents."""
    pos_by_accent: dict[tuple[int, ...], set[str]] = defaultdict(set)
    for tags, accents in readings:
        pos_by_accent[tuple(accents)].add(_feature_map(tags).get("upos", ""))
    if len(pos_by_accent) < 2:
        return False
    observed = set().union(*pos_by_accent.values())
    return UNRELIABLE_POS_SPLIT <= observed
