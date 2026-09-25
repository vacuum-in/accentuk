"""Classify the ambiguous surface by *why* a form carries more than one stress.

`stress_lookup` reports a form as ambiguous whenever it holds more than one
stress signature, but that single flag covers four situations with completely
different correct handling. Sending all of them to the contextual model wastes
latency on cases the model cannot learn and cases morphology already answers.

The discriminating evidence is the source dictionary's own structure. Its trie
stores readings as `(tags, accents)` records, and the shape of those records
says which situation applies:

    помилка   -> [([],                     [2, 4])]          one reading, two accents
    замок     -> [(NOUN Sing Nom Masc,     [2]),
                  (NOUN Sing Nom Masc,     [4]), ...]        same tags, different accents
    ведмедиці -> [(NOUN Sing Gen,          [2]),
                  (NOUN Plur Nom,          [4]), ...]        tags separate the accents

Accordingly:

`free_variation`
    A single reading carries several accents: the dictionary records both
    pronunciations as acceptable for one and the same reading. There is no
    right answer for a model to learn. `RESULTS.md` §5 recorded this failing
    once already -- 218 such senses were "unlearnable labels" and had to be
    excluded after the fact.

`grammatical`
    Readings differ in accent, but their tag sets separate them. Morphology
    decides, and the existing Stanza tag matching already does it. These look
    ambiguous in the wordlist only because it was stressed one word per line,
    where every token parses as an isolated nominative singular.

`homograph`
    Some tag set occurs with two different accents, so morphology cannot
    decide. This is the real model tier.

`artifact`
    Three or more accents under one tag set. `on_ambiguity=all` unions accent
    positions across readings, and beyond two this reflects over-merging rather
    than genuine n-way ambiguity (`березинська́`, stressed on the final
    syllable, is not plausible Ukrainian).

Known limitation, measured rather than assumed: the trie also uses the
single-record multi-accent notation for toponym and surname pairs that are
genuine homographs (`Бе́рестове`/`Берестове́` are two different villages).
`proper_noun_suspect` marks those for review instead of silently collapsing
them.
"""

from __future__ import annotations

import json
import unicodedata
from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

FREE_VARIATION = "free_variation"
GRAMMATICAL = "grammatical"
HOMOGRAPH = "homograph"
ARTIFACT = "artifact"
NOT_IN_SOURCE = "not_in_source"
PRIMARY_LEXICON = "primary_lexicon"

#: Features whose value Stanza predicts reliably enough to route on. `upos`
#: is deliberately absent: the readings it separates here are overwhelmingly
#: patronymic (NOUN) versus surname (PROPN), which is a naming distinction
#: rather than a morphological one, and both surface as proper names in running
#: text. Forms separated only by `upos` are reported so the weak claim stays
#: visible instead of being silently counted as solved.
RELIABLE_SEPARATING_FEATURES = frozenset({"Case", "Number", "Gender", "Animacy"})

#: Parts of speech whose split is not safely routable even when the tag sets
#: separate the accents. A noun and a verb sharing a spelling
#: (ко́си "braids" / коси́ "mow!", го́ри "mountains" / гори́ "burn!") are almost
#: always a noun beside an imperative, and Stanza tags the imperative as a
#: nominative plural noun — identically to the real noun. The trie's tags do
#: separate them, so tag matching looks applicable, but it inherits the parse
#: error and silently returns the noun reading every time. Measured on 5,958
#: grammatical forms, 656 carry this split.
UNRELIABLE_POS_SPLIT = frozenset({"NOUN", "VERB"})

#: Base class that always reaches the contextual model; `needs_model` adds
#: grammatical forms that only `upos` separates.
MODEL_TIER = frozenset({HOMOGRAPH})

#: Above this many accents under one tag set, treat as over-merging.
ARTIFACT_ACCENT_THRESHOLD = 3

Reading = tuple[list[str], list[int]]


@dataclass(frozen=True)
class TriagedForm:
    form: str
    form_normalized: str
    triage_class: str
    stressed_forms: tuple[str, ...]
    readings: int
    tagsets: int
    distinct_accents: int
    multi_accent_readings: int
    proper_noun_suspect: bool
    separating_features: tuple[str, ...] = ()
    morphologically_resolvable: bool = False
    review_reason: str | None = None


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
    """True when a noun reading and a verb reading carry different accents.

    See `UNRELIABLE_POS_SPLIT`: Stanza does not distinguish a Ukrainian
    imperative from a homographic nominative plural, so routing these on
    morphology returns the noun reading regardless of context.
    """
    pos_by_accent: dict[tuple[int, ...], set[str]] = defaultdict(set)
    for tags, accents in readings:
        pos_by_accent[tuple(accents)].add(_feature_map(tags).get("upos", ""))
    if len(pos_by_accent) < 2:
        return False
    observed = set().union(*pos_by_accent.values())
    return UNRELIABLE_POS_SPLIT <= observed


def classify_readings(readings: Iterable[Reading]) -> tuple[str, dict[str, int]]:
    """Return the triage class and the shape counters that produced it."""
    materialized = [(list(tags), list(accents)) for tags, accents in readings]
    if not materialized:
        return NOT_IN_SOURCE, {"readings": 0, "tagsets": 0, "distinct_accents": 0,
                               "multi_accent_readings": 0, "accent_positions": 0}

    accents_by_tagset: dict[frozenset[str], set[tuple[int, ...]]] = defaultdict(set)
    multi_accent_readings = 0
    positions: set[int] = set()
    for tags, accents in materialized:
        if len(accents) > 1:
            multi_accent_readings += 1
        positions.update(accents)
        accents_by_tagset[frozenset(tags)].add(tuple(accents))

    shape = {
        "readings": len(materialized),
        "tagsets": len(accents_by_tagset),
        "distinct_accents": len({tuple(a) for _, a in materialized}),
        "multi_accent_readings": multi_accent_readings,
        "accent_positions": len(positions),
    }

    # A tag set is *contested* when it carries two different single-accent
    # readings: morphology cannot separate them, so only context can. This is
    # tested before free variation because a form can have both — `плачу` has
    # two distinct 1sg VERB readings (пла́чу "cry" / плачу́ "pay", different
    # verbs) *and* NOUN readings written with variant notation [3,5]. Checking
    # free variation first saw the variant notation and stopped, misfiling a
    # genuine semantic homograph as "either stress is fine".
    #
    # Multi-accent readings are excluded from the contest: `[2,9]` alongside
    # `[2]` under one tag set is one word with an optional stress
    # (ба́тьківщи́на), not two words.
    single_by_tagset = {
        tags: {a for a in accents if len(a) == 1}
        for tags, accents in accents_by_tagset.items()
    }
    if any(len(a) > 1 for a in single_by_tagset.values()):
        widest_single = max(len(a) for a in single_by_tagset.values())
        return (ARTIFACT if widest_single >= ARTIFACT_ACCENT_THRESHOLD else HOMOGRAPH), shape
    if multi_accent_readings:
        return FREE_VARIATION, shape
    if len(positions) < 2:
        # The source dictionary offers a single stress for this form, so it
        # cannot be what made the form ambiguous -- the second reading came
        # from the primary lexicon, whose ambiguity this classifier has no
        # evidence about. Calling it `grammatical` would assert that
        # morphology resolves it, which nothing here supports.
        return PRIMARY_LEXICON, shape
    widest = max(len(a) for a in accents_by_tagset.values())
    if widest >= ARTIFACT_ACCENT_THRESHOLD:
        return ARTIFACT, shape
    if widest > 1:
        return HOMOGRAPH, shape
    return GRAMMATICAL, shape


def is_proper_noun_suspect(triage_class: str, stressed_forms: Iterable[str]) -> bool:
    """Flag collapsed readings that may be distinct names rather than variants.

    Only meaningful for `free_variation`: elsewhere the readings are kept apart
    anyway, so capitalisation carries no decision.
    """
    if triage_class != FREE_VARIATION:
        return False
    return any(form[:1].isupper() for form in stressed_forms)


class StressDictionary:
    """Adapter over the source package's trie, isolated for testability."""

    def __init__(self) -> None:
        from ukrainian_word_stress.stressify_ import _load_dictionary

        self._trie = _load_dictionary()

    def readings(self, form: str) -> list[Reading]:
        from ukrainian_word_stress.stressify_ import (
            _parse_dictionary_value,
            _trie_value,
        )

        # `stress_lookup.form_normalized` is NFD; the trie is keyed precomposed.
        value = _trie_value(self._trie, unicodedata.normalize("NFC", form))
        if value is None:
            return []
        return [(list(tags), list(accents)) for tags, accents in _parse_dictionary_value(value[0])]


def triage_forms(
    forms: Iterable[tuple[str, tuple[str, ...]]],
    dictionary: StressDictionary | None = None,
) -> Iterator[TriagedForm]:
    """Classify `(form_normalized, stressed_forms)` pairs from `stress_lookup`."""
    source = dictionary if dictionary is not None else StressDictionary()
    for form_normalized, stressed_forms in forms:
        readings = source.readings(form_normalized)
        triage_class, shape = classify_readings(readings)
        suspect = is_proper_noun_suspect(triage_class, stressed_forms)
        materialized = list(readings)
        features = separating_features(materialized) if triage_class == GRAMMATICAL else set()
        resolvable = bool(features & RELIABLE_SEPARATING_FEATURES)
        if triage_class == GRAMMATICAL and has_unreliable_pos_split(materialized):
            resolvable = False
        reason: str | None = None
        if triage_class == NOT_IN_SOURCE:
            reason = "form absent from the source dictionary"
        elif triage_class == PRIMARY_LEXICON:
            reason = "source dictionary gives one stress; ambiguity comes from the primary lexicon"
        elif triage_class == ARTIFACT:
            reason = f"{shape['distinct_accents']} accents under one tag set"
        elif triage_class == GRAMMATICAL and not resolvable:
            if has_unreliable_pos_split(materialized):
                reason = (
                    "noun and verb readings carry different accents; Stanza tags a "
                    "Ukrainian imperative as a nominative plural noun, so morphology "
                    "returns the noun reading regardless of context"
                )
            else:
                reason = (
                    "readings separated only by "
                    f"{','.join(sorted(features)) or 'nothing'}, which Stanza does not "
                    "predict reliably; treat as needing context"
                )
        elif suspect:
            reason = "capitalised readings may be distinct names, not free variation"
        yield TriagedForm(
            form=unicodedata.normalize("NFC", form_normalized),
            form_normalized=form_normalized,
            triage_class=triage_class,
            stressed_forms=tuple(stressed_forms),
            readings=shape["readings"],
            tagsets=shape["tagsets"],
            distinct_accents=shape["distinct_accents"],
            multi_accent_readings=shape["multi_accent_readings"],
            proper_noun_suspect=suspect,
            separating_features=tuple(sorted(features)),
            morphologically_resolvable=resolvable,
            review_reason=reason,
        )


def is_free_variation_by_gloss(definitions: Iterable[str]) -> bool:
    """True when a form's candidates describe the *same* sense.

    `classify_readings` works from the source dictionary's tag structure, which
    reports "two readings, tags do not separate them" for both a genuine
    semantic homograph (за́мок/замо́к) and a single word with two attested
    stresses. The gloss settles which: if the candidates carry the same
    definition, there is one meaning and two pronunciations — free variation —
    however the tags looked.

    This matters because such a form is *unresolvable* by a gloss-scoring
    model: both candidates present it with identical `(context, gloss)` input,
    so the pick is a coin flip made at whatever confidence the scores happen to
    differ by. Measured on held-out data these were 32% of all high-margin
    errors, and 1,724 forms across 166 groups were affected — overwhelmingly
    toponym adjectives like андру́шівська/андруші́вська, both glossed "relating
    to Andrushivka".

    Sending them to the model is strictly worse than resolving them from the
    dictionary, because a wrong answer at high confidence defeats the
    abstention threshold that protects the rest of the traffic.
    """
    cleaned = [" ".join(d.split()) for d in definitions]
    present = [d for d in cleaned if d]
    if len(present) < 2:
        return False
    return len(set(present)) < len(present)


def read_ambiguous_surface(
    database_url: str, *, dataset_id: int | None = None
) -> list[tuple[str, tuple[str, ...]]]:
    """Read every form the active dataset reports as ambiguous.

    Ambiguity is defined exactly as the Go API defines it: more than one
    distinct stress signature for one `form_normalized`.
    """
    import psycopg

    with psycopg.connect(database_url) as connection:
        if dataset_id is None:
            row = connection.execute(
                "SELECT dataset_id FROM active_dataset WHERE singleton"
            ).fetchone()
            if row is None:
                raise RuntimeError("no active dataset to triage")
            dataset_id = int(row[0])
        rows = connection.execute(
            """
            SELECT form_normalized, array_agg(DISTINCT stressed_form ORDER BY stressed_form)
            FROM stress_lookup
            WHERE dataset_id = %s
            GROUP BY form_normalized
            HAVING count(DISTINCT stress_signature) > 1
            ORDER BY form_normalized
            """,
            (dataset_id,),
        ).fetchall()
    return [(str(form), tuple(str(value) for value in forms)) for form, forms in rows]


def write_surface(path: Path, triaged: Iterable[TriagedForm]) -> dict[str, int]:
    """Write the triaged surface as JSONL and return per-class counts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = defaultdict(int)
    derived: dict[str, int] = defaultdict(int)
    total = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in triaged:
            total += 1
            counts[row.triage_class] += 1
            if row.proper_noun_suspect:
                derived["proper_noun_suspect"] += 1
            if row.triage_class == GRAMMATICAL and not row.morphologically_resolvable:
                derived["grammatical_unresolvable"] += 1
            if needs_model(row):
                derived["model_tier"] += 1
            if needs_review(row):
                derived["needs_review"] += 1
            handle.write(json.dumps(asdict(row), ensure_ascii=False, sort_keys=True) + "\n")
    counts["total"] = total
    counts.update(derived)
    return dict(counts)


def needs_model(row: TriagedForm) -> bool:
    """True when only sentence context can decide this form's stress."""
    if row.triage_class == HOMOGRAPH:
        return True
    # Separated by a feature Stanza does not predict reliably: morphology is
    # not actually enough, so these belong with the homographs.
    return row.triage_class == GRAMMATICAL and not row.morphologically_resolvable


def needs_review(row: TriagedForm) -> bool:
    """True when no automatic decision is defensible and a human must look."""
    return (
        row.triage_class in {ARTIFACT, NOT_IN_SOURCE, PRIMARY_LEXICON}
        or row.proper_noun_suspect
    )


def write_manifest(surface_path: Path, counts: dict[str, int]) -> dict[str, object]:
    """Record the surface's content hash so downstream artifacts can pin it."""
    import hashlib

    digest = hashlib.sha256(surface_path.read_bytes()).hexdigest()
    manifest = {
        "surface": surface_path.name,
        "surface_sha256": digest,
        "counts": counts,
        "model_tier_rule": (
            "homograph, plus grammatical forms whose readings are separated only by "
            "features Stanza does not predict reliably (see needs_model)"
        ),
        "schema_version": 1,
    }
    manifest_path = surface_path.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest
