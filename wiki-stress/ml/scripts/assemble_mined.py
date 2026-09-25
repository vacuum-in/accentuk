"""Turn mined + labelled sentences into training rows.

`corpus.build_rows` already holds every acceptance rule the Wikipedia corpus was
built with — label present and not unclear, sense inside the form's group, cue
actually occurring in the sentence, blind verification agreeing, span intact,
signature applying cleanly to the surface. This reuses it unchanged so a
sentence mined from subtitles or UberText is admitted on exactly the same terms
as one mined from Wikipedia.

It emits the flat row shape the trainer reads (`silver_*.json`), so the output
merges with the existing corpus through `merge_corpus.py`.

The reason this path exists at all: generation is bounded by an API quota that
admits roughly one request in flight, and a generation call spends 900–3,200
output tokens to yield about seven usable sentences. A labelling call carries 40
sentences and emits a few hundred tokens, and mining itself costs no quota —
30,462 candidates came out of the subtitle corpus in ten minutes of CPU. For the
same quota, mining and labelling produce roughly twenty times the rows, and they
are natural sentences rather than invented ones.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any

from ukstress_ml import ambiguity, annotate, corpus


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, action="append", required=True)
    parser.add_argument("--label", type=Path, action="append", required=True)
    parser.add_argument("--verify", type=Path, action="append", default=[])
    parser.add_argument("--inventory", type=Path,
                        default=Path("output/ml/ambiguous_forms_glossed.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("output/ml/mined_labelled.json"))
    args = parser.parse_args()

    forms = {f.form: f for f in ambiguity.load(args.inventory)}
    candidates: list[dict[str, Any]] = []
    for path in args.candidates:
        with path.open(encoding="utf-8") as handle:
            candidates += [json.loads(line) for line in handle]

    labels: dict[str, dict[str, Any]] = {}
    for path in args.label:
        labels.update(annotate.parse_raw(path))
    verifications: dict[str, dict[str, Any]] = {}
    for path in args.verify:
        verifications.update(annotate.parse_raw(path))

    print(f"candidates={len(candidates):,} labels={len(labels):,} "
          f"verifications={len(verifications):,}", flush=True)

    report = corpus.AssemblyReport()
    quarantine: list[dict[str, Any]] = []
    rows = corpus.build_rows(candidates, labels, verifications, forms, report, quarantine)
    rows = corpus.deduplicate(rows, report, quarantine)

    flat = [{
        "sentence_id": row["sentence_id"],
        "sentence": row["sentence"],
        "start": row["start"],
        "end": row["end"],
        "form": row["form"],
        "surface": row["surface"],
        "group_id": row["group_id"],
        "gold_sense": row["sense_id"],
        "gold": sorted(c["sense_id"] for c in row["candidates"]).index(row["sense_id"]),
        "corpus": row.get("corpus"),
        "licence": row.get("licence"),
        "source_tier": row.get("source_tier"),
        "page_id": row.get("page_id"),
        "revision_id": row.get("revision_id"),
        "page_title": None,
    } for row in rows]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(flat, ensure_ascii=False), encoding="utf-8")

    senses = collections.defaultdict(set)
    for row in flat:
        senses[row["group_id"]].add(row["gold_sense"])
    multi = sum(1 for v in senses.values() if len(v) > 1)
    print(f"\nrows kept   : {len(flat):,}")
    print(f"groups      : {len(senses):,}  of which multi-sense: {multi:,}")
    print(f"quarantined : {len(quarantine):,}")
    print(json.dumps(report.__dict__, indent=2, default=str))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
