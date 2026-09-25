"""Measure the ranker on homographs with an independent judge.

The ranker labels every audio row the token classifier trains on, and its
precision on homographs has never been measurable: the lexicon offers two
readings and nothing says which was spoken. A second, independent judge
does: the LLM sense labeller that built the cross-encoder corpus reads the
sentence with the sense definitions and names the sense, and each sense
carries its stress signature. Where the two agree the row is almost surely
right; where they disagree, one of them is wrong, and a hundred of those
chunks by ear says which.

Rows come from a classifier corpus (already placed in their sentence and
already past the confidence gate — these are the rows that train), sampled
per form and per reading, books and YouTube apart.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path

from ukstress_ml import ambiguity, annotate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("output/ml/corpus/tok-v6"))
    parser.add_argument("--forms", type=Path, default=Path("output/ml/ambiguous_forms.jsonl"))
    parser.add_argument("--per-reading", type=int, default=8,
                        help="rows per (form, reading, source) at most")
    parser.add_argument("--max-forms", type=int, default=0, help="pilot: this many forms")
    parser.add_argument("--every-minority-youtube", action="store_true",
                        help="not a sample: every YouTube row of a minority reading, "
                             "so the corpus can keep only those the judge confirms "
                             "(the ear check put the ranker at ~66 pct there)")
    parser.add_argument("--raw", type=Path, default=Path("output/ml/raw/judge_ranker.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("output/ml/judge_ranker"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    forms = {f.form: f for f in ambiguity.load(args.forms) if f.complete
             # compounds and either-stress variants carry signatures the
             # classifier corpus does not use; they cannot be compared
             if all(":" not in c.signature and "|" not in c.signature for c in f.candidates)}
    signature_of = {form: {c.sense_id: c.signature for c in f.candidates} for form, f in forms.items()}

    # Rows by (form, gold, source); the sample keeps both readings and both
    # sources in view, so the number is by reading, not by the majority.
    pools: dict[tuple[str, str, str], list[dict]] = collections.defaultdict(list)
    majority: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for part in ("train", "dev", "test"):
        for line in (args.corpus / f"{part}.jsonl").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            form = row["form"].lower()
            if form not in forms or len(row["candidates"]) < 2:
                continue
            source = "youtube" if str(row.get("book", "")).startswith("youtube:") else "book"
            pools[(form, str(row["gold"]), source)].append(row)
            majority[form][str(row["gold"])] += 1
    rng = random.Random(args.seed)
    chosen_forms = sorted({k[0] for k in pools})
    if args.max_forms:
        rng.shuffle(chosen_forms)
        chosen_forms = sorted(chosen_forms[: args.max_forms])
    rows = []
    for key in sorted(pools):
        if key[0] not in chosen_forms:
            continue
        pool = pools[key]
        if args.every_minority_youtube:
            if key[2] != "youtube" or key[1] == majority[key[0]].most_common(1)[0][0]:
                continue
            pool = list(pool)
            for row in pool:
                rows.append({"sentence_id": f"{row['book']}:{row['doc']}:{row['paragraph']}:{row['start']}",
                             "form": key[0], "sentence": row["sentence"], "gold": key[1],
                             "source": key[2], "confidence": row.get("confidence"),
                             "reading": "minority", "book": row["book"],
                             "start": row["start"], "end": row["end"]})
            continue
        # a draw of its own per cell, so adding or dropping a form elsewhere
        # does not reshuffle every other cell (and orphan the saved verdicts)
        random.Random(f"{args.seed}:{key}").shuffle(pool)
        for row in pool[: args.per_reading]:
            rows.append({"sentence_id": f"{row['book']}:{row['doc']}:{row['paragraph']}:{row['start']}",
                         "form": key[0], "sentence": row["sentence"], "gold": key[1],
                         "source": key[2], "confidence": row.get("confidence"),
                         "reading": "majority" if key[1] == majority[key[0]].most_common(1)[0][0] else "minority",
                         "book": row["book"], "start": row["start"], "end": row["end"]})
    print(f"{len(rows):,} rows over {len(chosen_forms):,} forms "
          f"({sum(1 for r in rows if r['source'] == 'youtube'):,} YouTube)", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "rows.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    config = annotate.load_config()
    usage = annotate.run_job(job="judge", rows=rows, forms=forms,
                             deployment=annotate.LABEL_DEPLOYMENT, config=config,
                             raw_path=args.raw, batch_size=20, workers=args.workers,
                             prompt_builder=annotate.label_prompt, system=annotate.LABEL_SYSTEM)
    print(f"usage {usage.snapshot()}", flush=True)

    verdicts = annotate.parse_raw(args.raw)
    args.out.mkdir(parents=True, exist_ok=True)
    tally: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    by_form: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    disagreements = []
    for row in rows:
        v = verdicts.get(row["sentence_id"])
        if not v or v.get("unclear") or not v.get("sense_id"):
            tally[row["source"]]["unclear"] += 1
            continue
        judge = signature_of[row["form"]].get(str(v["sense_id"]))
        if judge is None:
            tally[row["source"]]["unmapped"] += 1
            continue
        agree = judge == row["gold"]
        bucket = "≥0.99" if (row["confidence"] or 0) >= 0.99 else "<0.99"
        tally[row["source"]]["agree" if agree else "disagree"] += 1
        tally[f"{row['source']} {bucket}"]["agree" if agree else "disagree"] += 1
        tally[f"{row['source']} {row['reading']} reading"]["agree" if agree else "disagree"] += 1
        tally[f"{row['reading']} reading"]["agree" if agree else "disagree"] += 1
        by_form[row["form"]]["agree" if agree else "disagree"] += 1
        if not agree:
            disagreements.append({**row, "judge": judge, "judge_confidence": v.get("confidence"),
                                  "cue": v.get("cue")})
    lines = ["| slice | n | agree | disagree | unclear |", "|---|---:|---:|---:|---:|"]
    for name in sorted(tally):
        t = tally[name]
        n = t["agree"] + t["disagree"]
        lines.append(f"| {name} | {n:,} | {t['agree'] / max(n, 1):.1%} | {t['disagree']:,} "
                     f"| {t['unclear'] + t['unmapped']:,} |")
    worst = sorted(by_form.items(), key=lambda kv: (-kv[1]["disagree"], kv[0]))[:40]
    lines += ["", "| form | agree | disagree |", "|---|---:|---:|"]
    lines += [f"| {f} | {c['agree']} | {c['disagree']} |" for f, c in worst]
    (args.out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    confirmed = [r["sentence_id"] for r in rows
                 if (v := verdicts.get(r["sentence_id"])) and not v.get("unclear")
                 and signature_of[r["form"]].get(str(v.get("sense_id"))) == r["gold"]]
    (args.out / "confirmed.txt").write_text("\n".join(confirmed) + "\n", encoding="utf-8")
    (args.out / "disagreements.jsonl").write_text(
        "\n".join(json.dumps(d, ensure_ascii=False) for d in disagreements) + "\n", encoding="utf-8")
    print("\n".join(lines[:12]))
    print(f"\n{len(disagreements):,} disagreements -> {args.out / 'disagreements.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
