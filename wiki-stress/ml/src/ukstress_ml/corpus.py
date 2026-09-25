"""Corpus assembly: validate, verify, deduplicate, balance, split, encode.

The target string is built by applying the sense's stress signature to the
*surface* token, so casing is preserved (``Вона`` -> ``Вона́``) and the target is
always the observed token plus one acute.
"""

from __future__ import annotations

import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ukstress.normalizer import (
    ACUTE,
    UKRAINIAN_VOWELS,
    canonical_stressed_form,
    lookup_key,
    stress_signature,
    validate_stress,
)

from ukstress_ml.ambiguity import AmbiguousForm

OPEN_MARK = "⟦"
CLOSE_MARK = "⟧"
_WS = re.compile(r"\s+")


class EncodingError(ValueError):
    """Raised when a row cannot be rendered into model input."""


def apply_signature(surface: str, signature: str) -> str:
    """Return ``surface`` with an acute at the vowel ordinals of ``signature``.

    Keeps the observed casing and spelling of the token rather than substituting
    the inventory's lemma form.
    """
    if ":" in signature:
        raise EncodingError("multi-token signature is not supported for a single token")
    ordinals = {int(part) for part in signature.split("|")}
    canonical = canonical_stressed_form(surface)
    if ACUTE in canonical:
        raise EncodingError("surface already carries an acute")

    output: list[str] = []
    vowel_ordinal = -1
    position = 0
    while position < len(canonical):
        character = canonical[position]
        output.append(character)
        position += 1
        if character.lower() not in UKRAINIAN_VOWELS:
            continue
        # `ї` is `і` plus a combining diaeresis; the acute belongs after the
        # letter's own marks, not between the base and its diaeresis.
        while position < len(canonical) and unicodedata.combining(canonical[position]):
            output.append(canonical[position])
            position += 1
        vowel_ordinal += 1
        if vowel_ordinal in ordinals:
            output.append(ACUTE)
    result = "".join(output)
    if result.count(ACUTE) != len(ordinals):
        raise EncodingError(f"signature {signature} does not fit {surface!r}")
    return result


def encode_source(sentence: str, start: int, end: int) -> str:
    if OPEN_MARK in sentence or CLOSE_MARK in sentence:
        raise EncodingError("sentence contains a reserved span marker")
    return f"{sentence[:start]}{OPEN_MARK}{sentence[start:end]}{CLOSE_MARK}{sentence[end:]}"


GLOSS_MARK = "⟨"
GLOSS_CLOSE = "⟩"


def encode_source_with_glosses(
    sentence: str,
    start: int,
    end: int,
    candidates: list[dict[str, Any]],
    *,
    seed_key: str,
) -> str:
    """Encode the span *and* what each candidate stress means.

    Without this the model cannot answer for a homograph it has not memorised:
    the source names two stress positions and nothing about what either one
    means, and 96% of this inventory is same-part-of-speech pairs whose
    stress-to-meaning mapping is purely lexical. Pairing each stressed form with
    its gloss turns an unseen group from a coin flip into a matching task.

    Candidate order is shuffled per sentence so position carries no signal.
    """
    marked = encode_source(sentence, start, end)
    order = sorted(candidates, key=lambda c: c["sense_id"])
    random.Random(seed_key).shuffle(order)
    blocks = [
        f"{GLOSS_MARK}{c['stressed']}{GLOSS_CLOSE} {c.get('definition') or ''}".strip()
        for c in order
    ]
    return f"{marked} {' '.join(blocks)}"


def decode_source(source: str) -> tuple[str, int, int]:
    """Recover ``(sentence, start, end)`` from an encoded source string."""
    start = source.index(OPEN_MARK)
    end = source.index(CLOSE_MARK)
    sentence = source[:start] + source[start + 1 : end] + source[end + 1 :]
    return sentence, start, end - 1


def _shingles(text: str, size: int = 5) -> set[str]:
    tokens = lookup_key(text).split()
    if len(tokens) < size:
        return {" ".join(tokens)}
    return {" ".join(tokens[i : i + size]) for i in range(len(tokens) - size + 1)}


@dataclass
class AssemblyReport:
    mined_rows: int = 0
    labelled: int = 0
    unclear: int = 0
    label_out_of_group: int = 0
    label_missing_cue: int = 0
    verified: int = 0
    verify_disagree: int = 0
    verify_unclear: int = 0
    rejected_span: int = 0
    rejected_signature: int = 0
    rejected_stress_validation: int = 0
    rejected_projection: int = 0
    rejected_marker: int = 0
    duplicates_removed: int = 0
    near_duplicates_removed: int = 0
    label_conflicts: int = 0
    balanced_out: int = 0
    accepted: int = 0
    senses_covered: int = 0
    groups_covered: int = 0
    quarantined: int = 0
    splits: dict[str, int] = field(default_factory=dict)
    rows_per_group: dict[str, int] = field(default_factory=dict)
    eligible_groups_natural: int = 0
    cross_split_duplicates_removed: int = 0


def build_rows(
    mined: list[dict[str, Any]],
    labels: dict[str, dict[str, Any]],
    verifications: dict[str, dict[str, Any]],
    forms: dict[str, AmbiguousForm],
    report: AssemblyReport,
    quarantine: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Validate and label mined rows, quarantining anything that fails."""
    rows: list[dict[str, Any]] = []
    report.mined_rows = len(mined)

    for item in mined:
        sentence_id = item["sentence_id"]
        label = labels.get(sentence_id)
        if label is None:
            continue
        report.labelled += 1

        def reject(rule: str, item: dict[str, Any] = item, label: dict[str, Any] = label) -> None:
            quarantine.append({**item, "rule": rule, "label": label})

        if label["unclear"]:
            report.unclear += 1
            reject("label_unclear")
            continue

        form = forms.get(item["form"])
        if form is None:
            reject("form_not_in_inventory")
            continue
        by_sense = {c.sense_id: c for c in form.candidates}
        candidate = by_sense.get(label["sense_id"])
        if candidate is None:
            report.label_out_of_group += 1
            reject("label_out_of_group")
            continue

        cue = (label.get("cue") or "").strip()
        if not cue or not _cue_present(cue, item["sentence"]):
            report.label_missing_cue += 1
            reject("label_cue_absent")
            continue

        verification = verifications.get(sentence_id)
        if verification is not None:
            if verification["unclear"]:
                report.verify_unclear += 1
                reject("verify_unclear")
                continue
            report.verified += 1
            if verification["sense_id"] != label["sense_id"]:
                report.verify_disagree += 1
                reject("verify_disagree")
                continue

        sentence = item["sentence"]
        start, end = item["start"], item["end"]
        surface = sentence[start:end]
        if surface != item["surface"]:
            report.rejected_span += 1
            reject("span_mismatch")
            continue

        try:
            target = apply_signature(surface, candidate.signature)
            source = encode_source(sentence, start, end)
        except EncodingError as error:
            if "marker" in str(error):
                report.rejected_marker += 1
                reject("reserved_marker")
            else:
                report.rejected_signature += 1
                reject("signature_does_not_fit")
            continue

        if validate_stress(target).classification == "rejected":
            report.rejected_stress_validation += 1
            reject("stress_validation")
            continue
        if lookup_key(target) != lookup_key(surface):
            report.rejected_projection += 1
            reject("unstressed_projection")
            continue

        rows.append(
            {
                "sentence_id": sentence_id,
                "source": source,
                "target": target,
                "sentence": sentence,
                "start": start,
                "end": end,
                "surface": surface,
                "form": item["form"],
                "group_id": item["group_id"],
                "sense_id": candidate.sense_id,
                "signature": stress_signature(target),
                "candidates": [
                    {"sense_id": c.sense_id, "signature": c.signature} for c in form.candidates
                ],
                "label_origin": (
                    "generated" if item.get("source_tier") == "generated" else "llm"
                ),
                "label_confidence": label.get("confidence"),
                "verified": verification is not None,
                "source_tier": item.get("source_tier", "wiki"),
                "corpus": item.get("corpus"),
                "licence": item.get("licence"),
                "page_id": item.get("page_id"),
                "revision_id": item.get("revision_id"),
            }
        )
    return rows


def _cue_present(cue: str, sentence: str) -> bool:
    """True when at least half the cue's words occur in the sentence."""
    sentence_words = set(lookup_key(sentence).split())
    cue_words = [w for w in lookup_key(cue).split() if len(w) > 2]
    if not cue_words:
        return False
    hits = sum(1 for word in cue_words if word in sentence_words)
    return hits >= max(1, len(cue_words) // 2)


def deduplicate(
    rows: list[dict[str, Any]], report: AssemblyReport, quarantine: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    by_text: dict[str, dict[str, Any]] = {}
    label_by_text: dict[str, str] = {}
    for row in sorted(rows, key=lambda r: r["sentence_id"]):
        key = lookup_key(_WS.sub(" ", row["sentence"]))
        if key in by_text:
            if label_by_text[key] != row["sense_id"]:
                report.label_conflicts += 1
                quarantine.append({**row, "rule": "conflicting_label"})
            report.duplicates_removed += 1
            continue
        by_text[key] = row
        label_by_text[key] = row["sense_id"]

    kept: list[dict[str, Any]] = []
    seen_shingles: dict[int, set[str]] = defaultdict(set)
    for row in by_text.values():
        shingles = _shingles(row["sentence"])
        pool = seen_shingles[row["group_id"]]
        if shingles and len(shingles & pool) / len(shingles) >= 0.8:
            report.near_duplicates_removed += 1
            quarantine.append({**row, "rule": "near_duplicate"})
            continue
        pool |= shingles
        kept.append(row)
    return kept


def balance(
    rows: list[dict[str, Any]], report: AssemblyReport, *, max_ratio: int = 4, seed: int = 7
) -> list[dict[str, Any]]:
    """Cap the majority/minority ratio inside each group."""
    rng = random.Random(seed)
    by_group: dict[int, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_group[row["group_id"]][row["sense_id"]].append(row)

    kept: list[dict[str, Any]] = []
    for group in by_group.values():
        smallest = min(len(v) for v in group.values())
        cap = max(smallest * max_ratio, 1)
        for sense_rows in group.values():
            if len(sense_rows) > cap:
                rng.shuffle(sense_rows)
                report.balanced_out += len(sense_rows) - cap
                sense_rows = sense_rows[:cap]
            kept.extend(sense_rows)
    return kept


def split(
    rows: list[dict[str, Any]],
    report: AssemblyReport,
    *,
    seed: int = 20260815,
    unseen_groups: int = 120,
    dev_fraction: float = 0.1,
    test_fraction: float = 0.1,
) -> dict[str, list[dict[str, Any]]]:
    rng = random.Random(seed)
    by_group: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_group[row["group_id"]].append(row)

    # Reserve whole groups that carry both senses, so the unseen set is a real
    # disambiguation task rather than a single-label one.
    eligible = sorted(
        gid
        for gid, group_rows in by_group.items()
        if len({r["sense_id"] for r in group_rows}) >= 2 and len(group_rows) >= 8
    )
    rng.shuffle(eligible)
    unseen = set(eligible[:unseen_groups])

    splits: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "dev": [],
        "test": [],
        "test_unseen": [],
    }
    for group_id, group_rows in sorted(by_group.items()):
        if group_id in unseen:
            splits["test_unseen"].extend(group_rows)
            continue
        by_sense: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in group_rows:
            by_sense[row["sense_id"]].append(row)
        for sense_rows in by_sense.values():
            sense_rows = sorted(sense_rows, key=lambda r: r["sentence_id"])
            rng.shuffle(sense_rows)
            total = len(sense_rows)
            n_dev = int(total * dev_fraction)
            n_test = int(total * test_fraction)
            if total >= 5:
                n_dev = max(1, n_dev)
                n_test = max(1, n_test)
            splits["dev"].extend(sense_rows[:n_dev])
            splits["test"].extend(sense_rows[n_dev : n_dev + n_test])
            splits["train"].extend(sense_rows[n_dev + n_test :])

    report.splits = {name: len(items) for name, items in splits.items()}
    return splits


def split_natural_eval(
    rows: list[dict[str, Any]],
    report: AssemblyReport,
    *,
    seed: int = 20260816,
    dev_fraction: float = 0.15,
    test_fraction: float = 0.25,
) -> dict[str, list[dict[str, Any]]]:
    """Evaluate only on mined sentences; train on everything else.

    The task is to pick the right form for a *known* word in a sentence, so no
    group is held out. What is held out is natural text: generated sentences
    were written to make their sense obvious, so scoring on them overstates
    real-world accuracy. Evaluation rows are real Wikipedia sentences the model
    never saw, drawn from groups that occur with more than one sense in natural
    text — the only rows where "read the context" is actually being tested.
    """
    rng = random.Random(seed)

    mined_senses: dict[int, set[str]] = defaultdict(set)
    for row in rows:
        if row["label_origin"] != "generated":
            mined_senses[row["group_id"]].add(row["sense_id"])
    eligible = {group for group, senses in mined_senses.items() if len(senses) > 1}

    by_sense: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    splits: dict[str, list[dict[str, Any]]] = {"train": [], "dev": [], "test_natural": []}

    for row in rows:
        if row["label_origin"] == "generated" or row["group_id"] not in eligible:
            splits["train"].append(row)
            continue
        by_sense[(row["group_id"], row["sense_id"])].append(row)

    for key in sorted(by_sense):
        sense_rows = sorted(by_sense[key], key=lambda r: r["sentence_id"])
        rng.shuffle(sense_rows)
        total = len(sense_rows)
        n_dev = int(total * dev_fraction)
        n_test = int(total * test_fraction)
        if total >= 4:
            n_dev = max(1, n_dev)
            n_test = max(1, n_test)
        # Never strand a sense with no training rows.
        if total - n_dev - n_test < 1:
            n_dev = min(n_dev, max(0, total - 1 - n_test))
        splits["dev"].extend(sense_rows[:n_dev])
        splits["test_natural"].extend(sense_rows[n_dev : n_dev + n_test])
        splits["train"].extend(sense_rows[n_dev + n_test :])

    _guard_cross_split_duplicates(splits, report)
    report.splits = {name: len(items) for name, items in splits.items()}
    report.eligible_groups_natural = len(eligible)
    return splits


def _guard_cross_split_duplicates(
    splits: dict[str, list[dict[str, Any]]], report: AssemblyReport
) -> None:
    """Drop any evaluation row that shares 4-gram structure with a training row."""
    train_shingles: dict[int, set[str]] = defaultdict(set)
    for row in splits["train"]:
        train_shingles[row["group_id"]] |= _shingles(row["sentence"], 4)

    removed = 0
    for name in ("dev", "test_natural"):
        kept: list[dict[str, Any]] = []
        for row in splits[name]:
            own = _shingles(row["sentence"], 4)
            pool = train_shingles[row["group_id"]]
            if own and len(own & pool) / len(own) >= 0.5:
                removed += 1
                continue
            kept.append(row)
        splits[name] = kept
    report.cross_split_duplicates_removed = removed


def write(
    splits: dict[str, list[dict[str, Any]]],
    quarantine: list[dict[str, Any]],
    report: AssemblyReport,
    manifest_extra: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, items in splits.items():
        with (output_dir / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
            for row in items:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (output_dir / "quarantine.jsonl").open("w", encoding="utf-8") as handle:
        for row in quarantine:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    all_rows = [row for items in splits.values() for row in items]
    report.accepted = len(all_rows)
    report.quarantined = len(quarantine)
    report.senses_covered = len({row["sense_id"] for row in all_rows})
    report.groups_covered = len({row["group_id"] for row in all_rows})
    counts = Counter(row["group_id"] for row in all_rows)
    report.rows_per_group = {
        "min": min(counts.values()) if counts else 0,
        "median": sorted(counts.values())[len(counts) // 2] if counts else 0,
        "max": max(counts.values()) if counts else 0,
    }

    manifest = {"report": vars(report), **manifest_extra}
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return manifest


def load_split(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def nfd(value: str) -> str:
    return unicodedata.normalize("NFD", value)
