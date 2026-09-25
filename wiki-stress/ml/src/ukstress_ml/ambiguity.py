"""The ambiguous surface: unstressed forms that collide across senses.

Only these forms need training data. Anything that separates orthographically is
already resolved by exact lookup in PostgreSQL.

This build is lemma-only. Inflected forms require a vendored accented
morphological dictionary (task 3.1); until one exists, transferring stress
across a paradigm would mean guessing on mobile-stress lexemes, which the
specification forbids.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from ukstress.normalizer import lookup_key

from ukstress_ml.inventory import Sense


@dataclass(frozen=True)
class Candidate:
    sense_id: str
    stressed: str
    signature: str
    pos: str
    definition: str
    priority: int | None
    #: Provenance of `definition`. An LLM-authored gloss is a *label
    #: definition* — the text the model is scored against — so it must stay
    #: distinguishable from a human-confirmed one all the way downstream
    #: rather than becoming silent ground truth. Defaults to empty so
    #: inventories written before glosses were authored still load.
    review_status: str = ""


@dataclass(frozen=True)
class AmbiguousForm:
    group_id: int
    form: str
    feats: str
    candidates: tuple[Candidate, ...]
    paradigm_source: str
    complete: bool


def is_variant_notation(sense: Sense) -> bool:
    """True when one token carries two acutes, meaning "either stress is fine".

    ``а́льфа-ро́зпад`` is a compound with one stress per token and is a valid
    label; ``ба́тьківщи́на`` records two acceptable stresses of a single token and
    is not a label any model can be scored against.
    """
    return ":" not in sense.signature and "|" in sense.signature


def build(senses: list[Sense]) -> tuple[list[AmbiguousForm], dict[str, object]]:
    by_group: dict[int, list[Sense]] = defaultdict(list)
    for sense in senses:
        by_group[sense.group_id].append(sense)

    forms: list[AmbiguousForm] = []
    separated = 0
    variant_senses = 0
    variant_groups: set[int] = set()
    for group_id in sorted(by_group):
        group = sorted(by_group[group_id], key=lambda s: s.sense_id)
        unlearnable = [s for s in group if is_variant_notation(s)]
        if unlearnable:
            variant_senses += len(unlearnable)
            variant_groups.add(group_id)
            continue

        by_form: dict[str, list[Sense]] = defaultdict(list)
        for sense in group:
            by_form[lookup_key(sense.stressed)].append(sense)

        for form in sorted(by_form):
            colliding = by_form[form]
            if len({s.signature for s in colliding}) < 2:
                separated += 1
                continue
            forms.append(
                AmbiguousForm(
                    group_id=group_id,
                    form=form,
                    feats="lemma",
                    candidates=tuple(
                        Candidate(
                            sense_id=s.sense_id,
                            stressed=s.stressed,
                            signature=s.signature,
                            pos=s.pos,
                            definition=s.definition,
                            priority=s.priority,
                        )
                        for s in colliding
                    ),
                    paradigm_source="lemma",
                    complete=all(s.complete for s in colliding),
                )
            )

    report: dict[str, object] = {
        "ambiguous_forms": len(forms),
        "groups": len({f.group_id for f in forms}),
        "forms_separated_orthographically": separated,
        "senses_excluded_variant_notation": variant_senses,
        "groups_excluded_variant_notation": len(variant_groups),
        "forms_with_complete_senses": sum(1 for f in forms if f.complete),
        "candidates_per_form": {
            str(size): sum(1 for f in forms if len(f.candidates) == size)
            for size in sorted({len(f.candidates) for f in forms})
        },
        "paradigm_coverage": "lemma-only; inflected forms pending a vendored accented dictionary",
    }
    return forms, report


def write(forms: list[AmbiguousForm], report: dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "ambiguous_forms.jsonl").open("w", encoding="utf-8") as handle:
        for form in forms:
            handle.write(json.dumps(asdict(form), ensure_ascii=False) + "\n")
    (output_dir / "ambiguous_forms.report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load(path: Path) -> list[AmbiguousForm]:
    forms: list[AmbiguousForm] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            data = json.loads(line)
            data["candidates"] = tuple(Candidate(**c) for c in data["candidates"])
            forms.append(AmbiguousForm(**data))
    return forms
