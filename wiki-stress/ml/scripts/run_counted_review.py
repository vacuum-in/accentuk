"""Turn the counted-form review into the list the morphology tier reads.

The review answers a question no dictionary and no corpus could. The orthoepic
dictionary confirms only positives — silence there filed `вікна` as having no
counted form, which is wrong. The labelled corpus is worse than silent: its
silver labels were produced by models reading that same dictionary, so mining
it for numeral phrases returns «три се́стри» and simply restates the assumption
under test.

So the verdicts are human judgement, and they are the one artefact here that no
pipeline can regenerate. They are written to `ml/data/` and tracked, unlike
everything under `output/`, which is reproducible by definition.

A row is included only when the reviewer picked a reading. `recheck` and a
verdict with no choice are left out and behave as they did before — the rule
does not fire — because an unanswered question is not an answer.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviews", type=Path,
                        default=Path("output/ml/.annotations/counted_candidates.json.jsonl"),
                        help="append-only review log from the dataset browser")
    parser.add_argument("--candidates", type=Path,
                        default=Path("output/ml/counted_candidates.json"))
    parser.add_argument("--tracked", type=Path,
                        default=Path("ml/data/counted_forms_reviewed.jsonl"),
                        help="where the decisions are kept under version control")
    parser.add_argument("--out", type=Path,
                        default=Path("output/ml/counted_forms_v2.json"),
                        help="the list COUNTED_FORM_LIST points at")
    args = parser.parse_args()

    candidates = json.loads(args.candidates.read_text(encoding="utf-8"))

    latest: dict[int, dict] = {}
    if args.reviews.is_file():
        for line in args.reviews.read_text(encoding="utf-8").splitlines():
            if line.strip():
                entry = json.loads(line)
                latest[int(entry["row"])] = entry

    # Decisions already tracked survive a lost or rotated annotation log.
    decided: dict[str, dict] = {}
    if args.tracked.is_file():
        for line in args.tracked.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                decided[row["form"]] = row

    tally: collections.Counter[str] = collections.Counter()
    for row, entry in latest.items():
        if row >= len(candidates):
            tally["row_out_of_range"] += 1
            continue
        candidate = candidates[row]
        choice = entry.get("proposed") or ""
        if choice not in ("counted", "nominative"):
            tally["unanswered"] += 1
            continue
        decided[candidate["form"]] = {
            "form": candidate["form"],
            "gender": candidate.get("gender", ""),
            "nominative_plural": candidate["nominative_plural"],
            "genitive_singular": candidate["genitive_singular"],
            "verdict": choice,
            "stressed": (candidate["genitive_singular"] if choice == "counted"
                         else candidate["nominative_plural"]),
            "source": "review",
        }
        tally[choice] += 1

    args.tracked.parent.mkdir(parents=True, exist_ok=True)
    args.tracked.write_text(
        "".join(json.dumps(decided[f], ensure_ascii=False) + "\n" for f in sorted(decided)),
        encoding="utf-8")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps([decided[f] for f in sorted(decided)], ensure_ascii=False, indent=1),
        encoding="utf-8")

    counted = sum(1 for row in decided.values() if row["verdict"] == "counted")
    print(f"reviewed rows read: {len(latest):,}")
    for name, count in tally.most_common():
        print(f"  {name:<18}{count:>6}")
    print(f"\ndecisions on file: {len(decided):,}  ({counted} counted, "
          f"{len(decided) - counted} nominative)")
    print(f"-> {args.tracked}   (tracked)")
    print(f"-> {args.out}       (COUNTED_FORM_LIST)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
