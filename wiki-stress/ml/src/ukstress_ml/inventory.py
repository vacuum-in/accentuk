"""Frozen homograph sense inventory built from the working CSV.

Every form comparison routes through ``ukstress.normalizer`` so that corpus
matching and dictionary lookup cannot disagree.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from ukstress.normalizer import (
    ACUTE,
    canonical_stressed_form,
    lookup_key,
    stress_signature,
    validate_stress,
)

_POS_RULES: tuple[tuple[str, str], ...] = (
    ("прикметник", "adjective"),
    ("дієприкметник", "participle"),
    ("дієприслівник", "converb"),
    ("дієслово", "verb"),
    ("множинний іменник", "noun"),
    ("іменник", "noun"),
    ("прислівник", "adverb"),
    ("займенник", "pronoun"),
    ("числівник", "numeral"),
    ("прийменник", "preposition"),
    ("сполучник", "conjunction"),
    ("частка", "particle"),
    ("вигук", "interjection"),
    ("прізвище", "proper_noun"),
    ("ім'я", "proper_noun"),
    ("імʼя", "proper_noun"),
    ("топонім", "proper_noun"),
    ("абревіатура", "abbreviation"),
)

# Register/usage labels that SUM prefixes onto a gloss. Split off so the gloss
# itself stays a definition rather than a mix of definition and usage label.
_REGISTER_RULES: tuple[str, ...] = (
    "Народно-поетичне слово, розмовне слово чи вираз",
    "Розмовне слово чи вираз",
    "Діалектне слово",
    "Застаріле слово",
    "Рідковживане слово",
    "Поетичне слово",
    "Народно-поетичне слово",
    "Зневажливе слово",
    "Пестливе слово",
    "Жаргонне слово",
    "Переносне значення",
    "Спеціальне значення",
)

_WS = re.compile(r"\s+")


@dataclass(frozen=True)
class Sense:
    """One stress realization of one ambiguous spelling."""

    sense_id: str
    group_id: int
    spelling: str
    stressed: str
    signature: str
    pos: str
    gram_style: str
    definition: str
    register: str
    contrast: str
    comment: str
    priority: int | None
    merged_definitions: tuple[str, ...] = ()
    review_status: str = "source"
    field_sources: dict[str, str] = field(default_factory=dict)

    @property
    def complete(self) -> bool:
        return bool(self.definition.strip())


def _clean(value: str | None) -> str:
    return _WS.sub(" ", (value or "").strip())


def parse_definition(raw: str | None) -> tuple[str, str]:
    """Return ``(definition, register)`` from a raw CSV definition cell.

    2 419 of the 4 829 source rows store a Python list rendered as a string.
    Those are parsed back into real glosses; the leading register label, when
    present, is separated so it does not read as part of the meaning.
    """
    text = _clean(raw)
    if not text:
        return "", ""

    if text.startswith("["):
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            parsed = None
        if isinstance(parsed, list):
            glosses = [_clean(str(item)) for item in parsed if _clean(str(item))]
            text = " ".join(glosses[:2])

    register = ""
    for label in _REGISTER_RULES:
        if text.startswith(label):
            register = label
            text = _clean(text[len(label) :].lstrip(" ,.;:"))
            break
    return text, register


def part_of_speech(gram_style: str) -> str:
    lowered = gram_style.casefold()
    for needle, code in _POS_RULES:
        if needle.casefold() in lowered:
            return code
    return "unknown"


def _sense_label(index: int) -> str:
    letters = "abcdefghijklmnopqrstuvwxyz"
    return letters[index] if index < len(letters) else f"x{index}"


def load_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


@dataclass
class InventoryReport:
    source_rows: int = 0
    groups_in: int = 0
    senses_out: int = 0
    groups_out: int = 0
    collapsed_senses: int = 0
    collapsed_groups: int = 0
    groups_dropped_single_stress: int = 0
    rows_rejected_invalid_stress: int = 0
    senses_incomplete: int = 0
    groups_with_incomplete_sense: int = 0
    definitions_parsed_from_list: int = 0
    registers_extracted: int = 0
    pos_distribution: dict[str, int] = field(default_factory=dict)
    same_pos_groups: int = 0
    cross_pos_groups: int = 0
    senses_per_group: dict[str, int] = field(default_factory=dict)
    rejected_examples: list[dict[str, str]] = field(default_factory=list)


def build(csv_path: Path) -> tuple[list[Sense], InventoryReport]:
    """Build the frozen sense list and its report from the working CSV."""
    rows = load_rows(csv_path)
    report = InventoryReport(source_rows=len(rows))

    by_group: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_group[int(row["group_id"])].append(row)
    report.groups_in = len(by_group)

    senses: list[Sense] = []
    for group_id in sorted(by_group):
        group_rows = by_group[group_id]
        spelling = lookup_key(group_rows[0]["spelling"])

        # Collapse rows that share a stressed form: two meanings, one
        # pronunciation, therefore one label a stress model could ever predict.
        merged: dict[str, dict[str, object]] = {}
        for row in group_rows:
            stressed = canonical_stressed_form(row["stressed_word"])
            if validate_stress(stressed).classification == "rejected":
                report.rows_rejected_invalid_stress += 1
                if len(report.rejected_examples) < 20:
                    report.rejected_examples.append(
                        {"group_id": str(group_id), "stressed_word": row["stressed_word"]}
                    )
                continue
            if lookup_key(stressed) != spelling:
                report.rows_rejected_invalid_stress += 1
                if len(report.rejected_examples) < 20:
                    report.rejected_examples.append(
                        {
                            "group_id": str(group_id),
                            "stressed_word": row["stressed_word"],
                            "reason": "unstressed projection differs from spelling",
                        }
                    )
                continue

            raw_definition = row.get("definition")
            definition, register = parse_definition(raw_definition)
            if _clean(raw_definition).startswith("["):
                report.definitions_parsed_from_list += 1
            if register:
                report.registers_extracted += 1

            key = stressed.casefold()
            if key in merged:
                bucket = merged[key]
                if definition:
                    extras = bucket["merged_definitions"]
                    assert isinstance(extras, list)
                    extras.append(definition)
                    if not bucket["definition"]:
                        bucket["definition"] = definition
                report.collapsed_senses += 1
                continue

            priority_raw = _clean(row.get("priority"))
            merged[key] = {
                "stressed": stressed,
                "definition": definition,
                "register": register,
                "gram_style": _clean(row.get("gram_style")),
                "comment": _clean(row.get("comment")),
                "extra_text": _clean(row.get("extra_text")),
                "priority": int(priority_raw) if priority_raw.isdigit() else None,
                "merged_definitions": [],
            }

        if len(merged) < 2:
            if len(merged) == 1:
                report.groups_dropped_single_stress += 1
            continue
        if len(merged) < len({lookup_key(r["stressed_word"]) for r in group_rows}) or any(
            bucket["merged_definitions"] for bucket in merged.values()
        ):
            report.collapsed_groups += 1

        for index, key in enumerate(sorted(merged)):
            bucket = merged[key]
            stressed = str(bucket["stressed"])
            gram_style = str(bucket["gram_style"])
            definition = str(bucket["definition"])
            extras = bucket["merged_definitions"]
            assert isinstance(extras, list)
            senses.append(
                Sense(
                    sense_id=f"{group_id}.{_sense_label(index)}",
                    group_id=group_id,
                    spelling=spelling,
                    stressed=stressed,
                    signature=stress_signature(stressed),
                    pos=part_of_speech(gram_style),
                    gram_style=gram_style,
                    definition=definition,
                    register=str(bucket["register"]),
                    contrast="",
                    comment=str(bucket["comment"]),
                    priority=bucket["priority"],  # type: ignore[arg-type]
                    merged_definitions=tuple(extras),
                    review_status="source" if definition else "incomplete",
                    field_sources={"definition": "csv", "stressed": "csv"},
                )
            )

    groups_out = {sense.group_id for sense in senses}
    report.senses_out = len(senses)
    report.groups_out = len(groups_out)
    report.senses_incomplete = sum(1 for sense in senses if not sense.complete)
    report.groups_with_incomplete_sense = len(
        {sense.group_id for sense in senses if not sense.complete}
    )
    report.pos_distribution = dict(Counter(sense.pos for sense in senses).most_common())

    per_group: dict[int, list[Sense]] = defaultdict(list)
    for sense in senses:
        per_group[sense.group_id].append(sense)
    report.senses_per_group = {
        str(size): count
        for size, count in sorted(Counter(len(v) for v in per_group.values()).items())
    }
    for group_senses in per_group.values():
        if len({sense.pos for sense in group_senses}) == 1:
            report.same_pos_groups += 1
        else:
            report.cross_pos_groups += 1
    return senses, report


def content_hash(senses: list[Sense]) -> str:
    digest = hashlib.sha256()
    for sense in sorted(senses, key=lambda s: s.sense_id):
        payload = json.dumps(
            [sense.sense_id, sense.spelling, sense.stressed, sense.signature, sense.definition],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        digest.update(payload.encode())
    return digest.hexdigest()


def write(senses: list[Sense], report: InventoryReport, output_dir: Path) -> dict[str, object]:
    """Write the frozen inventory, its report, and a manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)
    inventory_path = output_dir / "inventory_v1.jsonl"
    with inventory_path.open("w", encoding="utf-8") as handle:
        for sense in sorted(senses, key=lambda s: (s.group_id, s.sense_id)):
            handle.write(json.dumps(asdict(sense), ensure_ascii=False) + "\n")

    manifest: dict[str, object] = {
        "artifact": "inventory_v1",
        "content_hash": content_hash(senses),
        "created_at": datetime.now(UTC).isoformat(),
        "normalization_version": "1",
        "acute": f"U+{ord(ACUTE):04X}",
        "paradigm_source": "lemma-only (no morphological dictionary vendored yet)",
        "report": asdict(report),
    }
    (output_dir / "inventory_v1.manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def load(path: Path) -> list[Sense]:
    senses: list[Sense] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            data = json.loads(line)
            data["merged_definitions"] = tuple(data.get("merged_definitions", ()))
            senses.append(Sense(**data))
    return senses


def normalized_nfd(value: str) -> str:
    return unicodedata.normalize("NFD", value)
