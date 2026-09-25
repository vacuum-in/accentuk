"""Close the gap between the gloss inventory and the training corpus.

`ambiguous_forms_glossed.jsonl` has no candidate list for 2,263 of the corpus's
7,079 forms, so training silently drops **171,245 of 463,867 rows (36.9%)**.
That is why both arms of the exemplar ablation scored below the shipped model
and why neither was a deployment candidate.

Almost none of it is missing data. The serving manifest already carries
candidates for 2,111 of those 2,263 forms — 157,894 rows, 92% of the gap. The
two artifacts hold the same information in different shapes:

    manifest   {form: {group_id, candidates: [{sense_id, signature,
                                               stressed, definition}]}}
    inventory  {form, group_id, feats, paradigm_source, complete,
                candidates: [{..., pos, priority, review_status}]}

So this is a format merge, not a labelling campaign. The fields the manifest
lacks — `pos`, `priority`, `feats`, `paradigm_source` — are metadata that
training never reads; `candidate_glosses` uses `sense_id`, `stressed` and
`definition`, and sorts by `sense_id`. They are filled with explicit unknowns
rather than guesses, and every merged form is marked `paradigm_source:
"serving_manifest"` so it can be told apart later.

The residue is genuine: 152 forms and 13,351 rows have no gloss in either
artifact — mostly function words (`була`, `які`, `кілька`) that the trie calls
ambiguous but that carry no sense entry anywhere. Those stay out.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_inventory(path: Path) -> dict[str, dict[str, Any]]:
    out = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            out[row["form"]] = row
    return out


def manifest_entry(form: str, entry: dict[str, Any]) -> dict[str, Any]:
    """Reshape one manifest form into an inventory record."""
    return {
        "form": form,
        "group_id": entry.get("group_id"),
        "feats": "",
        "paradigm_source": "serving_manifest",
        "complete": True,
        "candidates": [
            {
                "sense_id": c["sense_id"],
                "stressed": c["stressed"],
                "signature": c["signature"],
                "definition": c.get("definition", ""),
                # Not carried by the manifest and not read during training.
                # Recorded as unknown rather than invented.
                "pos": "unknown",
                "priority": None,
                "review_status": "",
            }
            for c in sorted(entry["candidates"], key=lambda c: c["sense_id"])
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path,
                        default=Path("output/ml/ambiguous_forms_glossed.jsonl"))
    parser.add_argument("--manifest", type=Path,
                        default=Path("output/ml/serving_manifest_v22.json"))
    parser.add_argument("--corpus", type=Path,
                        default=Path("output/ml/silver_mined_v10.json"))
    parser.add_argument("--out", type=Path,
                        default=Path("output/ml/ambiguous_forms_merged.jsonl"))
    args = parser.parse_args(argv)

    inventory = load_inventory(args.inventory)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))["forms"]
    corpus = json.loads(args.corpus.read_text(encoding="utf-8"))

    needed: dict[str, int] = {}
    for row in corpus:
        needed[row["form"]] = needed.get(row["form"], 0) + 1

    added = recovered = 0
    for form, rows in needed.items():
        if form in inventory or form not in manifest:
            continue
        inventory[form] = manifest_entry(form, manifest[form])
        added += 1
        recovered += rows

    residue = {f: n for f, n in needed.items() if f not in inventory}
    with args.out.open("w", encoding="utf-8") as handle:
        for form in sorted(inventory):
            handle.write(json.dumps(inventory[form], ensure_ascii=False) + "\n")

    covered = sum(n for f, n in needed.items() if f in inventory)
    print(f"corpus            {len(corpus):,} rows across {len(needed):,} forms")
    print(f"added from manifest  {added:,} forms, recovering {recovered:,} rows")
    print(f"still uncovered      {len(residue):,} forms, {sum(residue.values()):,} rows")
    print(f"coverage             {100 * covered / len(corpus):.1f}% of rows "
          f"-> {args.out}")
    if residue:
        top = sorted(residue.items(), key=lambda kv: -kv[1])[:10]
        print("  largest residue:  " + ", ".join(f"{f} ({n})" for f, n in top))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
