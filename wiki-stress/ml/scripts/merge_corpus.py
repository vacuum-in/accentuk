"""Merge generated sentences into the training corpus.

Generated rows arrive with what generation knows — the sentence, the span, the
form, and the sense it was written for. The trainer additionally expects the
row shape the mined corpus has, so this fills in the provenance fields and the
stable id, and refuses to invent anything it cannot derive.

Two rules, both learned the expensive way:

* Provenance is **carried through, never assumed**. An added row keeps its own
  `corpus`, `licence` and `source_tier`; only rows that arrive without them
  fall back to synthetic. Stamping every addition `generated/synthetic` was
  wrong in both directions: it labelled 32,629 sentences mined from
  OpenSubtitles and Malyuk as synthetic, which misstates their licence in any
  export, and it would equally have hidden generated text inside a corpus
  claiming a Wikipedia licence.
* Deduplication is on the sentence text across *all* sources. Generation asked
  several models for the same (form, sense), and an identical sentence
  appearing twice would be counted twice toward a sense's floor while teaching
  nothing new.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
from typing import Any

from ukstress_ml import ambiguity


def sentence_id(sentence: str, form: str, start: int) -> str:
    digest = hashlib.sha256(f"{sentence}\x00{form}\x00{start}".encode()).hexdigest()
    return f"gen:{digest[:24]}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path,
                        default=Path("output/ml/silver_plus_generated_v2.json"))
    # `action="append"` appends to its default rather than replacing it, so a
    # default list here means an explicit --add silently processes the default
    # file as well. Resolve the fallback after parsing instead.
    parser.add_argument("--add", type=Path, action="append", default=None)
    parser.add_argument("--inventory", type=Path,
                        default=Path("output/ml/ambiguous_forms_glossed.jsonl"))
    parser.add_argument("--out", type=Path,
                        default=Path("output/ml/silver_balanced_v3.json"))
    args = parser.parse_args()

    forms = {f.form: f for f in ambiguity.load(args.inventory)}
    rows: list[dict[str, Any]] = json.loads(args.base.read_text(encoding="utf-8"))
    seen = {row["sentence"] for row in rows}
    added = 0
    skipped: collections.Counter[str] = collections.Counter()

    for path in (args.add or [Path("output/ml/generated_balanced.json")]):
        if not path.exists():
            continue
        for row in json.loads(path.read_text(encoding="utf-8")):
            sentence = row["sentence"]
            if sentence in seen:
                skipped["duplicate_sentence"] += 1
                continue
            form = forms.get(row["form"])
            if form is None:
                skipped["form_not_in_inventory"] += 1
                continue
            senses = [c.sense_id for c in sorted(form.candidates, key=lambda c: c.sense_id)]
            if row["gold_sense"] not in senses:
                skipped["sense_not_in_form"] += 1
                continue
            seen.add(sentence)
            added += 1
            rows.append({
                "sentence_id": sentence_id(sentence, row["form"], row["start"]),
                "sentence": sentence,
                "start": row["start"],
                "end": row["end"],
                "form": row["form"],
                "surface": sentence[row["start"]:row["end"]],
                "group_id": row["group_id"],
                "gold_sense": row["gold_sense"],
                "gold": senses.index(row["gold_sense"]),
                "corpus": row.get("corpus") or "generated",
                "licence": row.get("licence") or "synthetic",
                "source_tier": row.get("source_tier") or "generated",
                "page_id": row.get("page_id"),
                "revision_id": row.get("revision_id"),
                "page_title": row.get("page_title"),
            })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")

    have: dict[int, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    for row in rows:
        have[row["group_id"]][row["gold_sense"]] += 1
    bands: collections.Counter[str] = collections.Counter()
    for form in forms.values():
        senses = {c.sense_id for c in form.candidates}
        if len({c.signature for c in form.candidates}) < 2:
            continue
        counts = have.get(form.group_id, collections.Counter())
        total = sum(counts.values())
        if not total:
            bands["no data"] += 1
        elif any(counts.get(s, 0) == 0 for s in senses):
            bands["a sense with zero rows"] += 1
        elif min(counts.get(s, 0) for s in senses) / total < 0.20:
            bands["minority under 20%"] += 1
        else:
            bands["minority at or above 20%"] += 1

    print(f"base rows : {len(rows) - added:>8,}")
    print(f"added     : {added:>8,}")
    print(f"total     : {len(rows):>8,}  -> {args.out}")
    for reason, count in skipped.most_common():
        print(f"  skipped {count:>7,}  {reason}")
    print("\nform-level sense balance after merge:")
    total_forms = sum(bands.values())
    for band, count in bands.most_common():
        print(f"  {band:<26}{count:>7,}  {count / max(total_forms, 1):>6.1%}")


if __name__ == "__main__":
    main()
