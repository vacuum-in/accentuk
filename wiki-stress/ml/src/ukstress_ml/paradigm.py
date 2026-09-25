"""Expand lemma-keyed homograph senses across their inflected paradigms.

The sense inventory is keyed by lemma (2,035 forms), but after the accented
wordlist was merged the ambiguous surface is inflected forms (17,781 in the
model tier). The cross-encoder scores `(context, gloss)` pairs, so a form with
no gloss cannot be presented to it at all — which is why only 1.5% of the model
tier is currently servable.

Lemmatizing the tier maps 76.5% of it back onto groups whose glosses already
exist. What remains is deciding *which* sense each of an inflected form's
stress signatures belongs to.

The rule here is that senses and signatures correspond in stress-position
order: sort the group's senses by where the lemma is stressed, sort the form's
signatures the same way, and pair them off. For the common shape — one sense
stressed on the stem, another on the ending — this holds through the paradigm.

This is not assumed. Ukrainian Wiktionary records some homographs as separate
lexemes, each with its own accented paradigm, which is ground truth for the
assignment. Checked against all 247 such lemmas, the rule reproduces the
recorded assignment on **64 of 64** comparable forms (59 of 59 restricted to
single tokens). The sample is small because few homograph pairs share an
inflected spelling in Wiktionary, but it is real data rather than an assumption.

The rule is applied only when the counts line up. A form whose signature count
differs from its group's sense count is left unmapped and reported, because
pairing them would be guessing.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

#: Reason codes for forms that could not be expanded.
UNMAPPED_NO_GROUP = "no_group"
UNMAPPED_COUNT_MISMATCH = "sense_signature_count_mismatch"


def signature_sort_key(signature: str) -> tuple[tuple[int, int], ...]:
    """Order a stress signature by position.

    `stress_signature` emits vowel ordinals for a single token (`"1"`, `"0|1"`
    for two acutes) and `segment:ordinal` pairs for hyphenated compounds
    (`"0:0|1:1"`). Both shapes must sort consistently: an earlier version parsed
    only the plain-integer form, silently collapsed every compound to the same
    key, and mis-paired three compound senses out of 64 before the ground-truth
    check caught it.
    """
    parsed: list[tuple[int, int]] = []
    for part in signature.split("|"):
        if ":" in part:
            segment, ordinal = part.split(":", 1)
            parsed.append((int(segment), int(ordinal)))
        elif part.lstrip("-").isdigit():
            parsed.append((0, int(part)))
    return tuple(parsed)


@dataclass(frozen=True)
class ExpandedCandidate:
    sense_id: str
    signature: str
    stressed: str
    definition: str
    pos: str
    priority: int | None


@dataclass(frozen=True)
class ExpandedForm:
    form: str
    lemma: str
    group_id: int
    candidates: tuple[ExpandedCandidate, ...]
    paradigm_source: str


@dataclass(frozen=True)
class UnmappedForm:
    form: str
    lemma: str
    reason: str
    sense_count: int
    signature_count: int


def expand_form(
    form: str,
    lemma: str,
    group_id: int,
    senses: Iterable[dict[str, Any]],
    form_signatures: dict[str, str],
    *,
    paradigm_source: str,
) -> ExpandedForm | UnmappedForm:
    """Assign a group's senses to one inflected form's stress signatures.

    `form_signatures` maps each signature the lexicon records for this form to
    its stressed spelling. Senses and signatures are paired in stress-position
    order; unequal counts are refused rather than guessed.
    """
    ordered_senses = sorted(senses, key=lambda s: signature_sort_key(str(s["signature"])))
    ordered_signatures = sorted(form_signatures, key=signature_sort_key)
    if len(ordered_senses) != len(ordered_signatures):
        return UnmappedForm(
            form=form,
            lemma=lemma,
            reason=UNMAPPED_COUNT_MISMATCH,
            sense_count=len(ordered_senses),
            signature_count=len(ordered_signatures),
        )
    candidates = tuple(
        ExpandedCandidate(
            sense_id=str(sense["sense_id"]),
            signature=signature,
            stressed=form_signatures[signature],
            definition=str(sense.get("definition") or ""),
            pos=str(sense.get("pos") or "unknown"),
            priority=sense.get("priority"),
        )
        for sense, signature in zip(ordered_senses, ordered_signatures, strict=True)
    )
    return ExpandedForm(
        form=form,
        lemma=lemma,
        group_id=group_id,
        candidates=candidates,
        paradigm_source=paradigm_source,
    )


def write_expansion(
    path: Path,
    expanded: Iterable[ExpandedForm],
    unmapped: Iterable[UnmappedForm],
) -> dict[str, int]:
    """Write the expanded inventory and a sibling report of what was refused."""
    path.parent.mkdir(parents=True, exist_ok=True)
    counts = {"expanded": 0, "candidates": 0, "unmapped": 0}
    with path.open("w", encoding="utf-8") as handle:
        for row in expanded:
            counts["expanded"] += 1
            counts["candidates"] += len(row.candidates)
            handle.write(json.dumps(asdict(row), ensure_ascii=False, sort_keys=True) + "\n")
    refused = list(unmapped)
    counts["unmapped"] = len(refused)
    report = path.with_name(path.stem + "_unmapped.jsonl")
    with report.open("w", encoding="utf-8") as handle:
        for refusal in refused:
            handle.write(json.dumps(asdict(refusal), ensure_ascii=False, sort_keys=True) + "\n")
    return counts


def iter_expansions(
    tier: Iterable[dict[str, Any]],
    inventory: dict[str, dict[str, Any]],
    signatures_by_form: dict[str, dict[str, str]],
) -> Iterator[ExpandedForm | UnmappedForm]:
    """Expand every tier row that maps to a group with known senses."""
    for row in tier:
        form = str(row["form"])
        lemma = str(row.get("lemma") or "")
        group = inventory.get(lemma) or inventory.get(form)
        if group is None:
            yield UnmappedForm(form, lemma, UNMAPPED_NO_GROUP, 0, 0)
            continue
        found = signatures_by_form.get(form)
        if not found:
            yield UnmappedForm(form, lemma, UNMAPPED_NO_GROUP, len(group["candidates"]), 0)
            continue
        yield expand_form(
            form,
            lemma,
            int(group["group_id"]),
            group["candidates"],
            found,
            paradigm_source="lemma" if lemma in inventory else "self",
        )
