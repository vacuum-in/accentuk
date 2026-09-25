"""Build the token-classifier corpus from the verified sweeps.

One row per (sentence, span): what was said, where the word sits in it, which
readings the lexicon offers, and which one the speakers chose. That is all a
token classifier needs, and it is what no other source in this project
supplies — the cross-encoder's corpus carries glosses, and a grammatical
heteronym has one gloss for both readings.

The gates and their order are from PLAN_ACCENTOR.md §4. Every count is written
to the manifest, because a dataset whose filtering cannot be audited is one
this project has already been burned by.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures as futures
import csv
import json
import random
import re
import sys
import unicodedata
import urllib.request
from pathlib import Path

VOWELS = "аеєиіїоуюяй"
WORD = re.compile(r"[А-Яа-яЇїІіЄєҐґ'’-]+")
PREPOSITIONS = {
    "до", "у", "в", "на", "за", "під", "над", "про", "від", "од", "біля", "для",
    "без", "при", "через", "перед", "між", "із", "з", "зі", "повз", "поза",
    "попід", "коло", "після", "проти", "серед", "крім", "щодо",
}
CLITICS = {
    "мене", "тебе", "себе", "нього", "неї", "них", "нас", "вас", "мною",
    "тобою", "собою", "ньому", "нім", "нею", "ними",
}


def load_sweep(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip() and '"form"' in line]
    return json.loads(path.read_text(encoding="utf-8"))


def is_clitic(sentence: str, form: str) -> bool:
    lowered = form.lower()
    if lowered not in CLITICS:
        return False
    words = [w.lower() for w in WORD.findall(sentence)]
    return any(w == lowered and i and words[i - 1] in PREPOSITIONS
               for i, w in enumerate(words))


def ask(endpoint: str, text: str) -> list[dict]:
    payload = json.dumps({"text": text}).encode("utf-8")
    request = urllib.request.Request(
        endpoint, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read())["tokens"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweeps", nargs="+", type=Path, required=True)
    parser.add_argument("--corpus", type=Path,
                        default=Path("/home/devops/audiotostress/data/cv-corpus-26.0-2026-06-12/uk"))
    parser.add_argument("--endpoint", default="http://localhost:8080/v1/stress")
    parser.add_argument("--out", type=Path, default=Path("output/ml/corpus/tok-v1"))
    parser.add_argument("--min-confidence", type=float, default=0.99)
    parser.add_argument("--pair-confidence", type=float, default=0.95)
    parser.add_argument("--per-form-cap", type=int, default=60)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}

    sentences: dict[str, str] = {}
    speakers: dict[str, str] = {}
    split_of: dict[str, str] = {}
    with (args.corpus / "validated.tsv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t", quoting=csv.QUOTE_NONE):
            sentences[row["path"]] = row["sentence"]
            speakers[row["path"]] = row["client_id"]
    for name in ("train", "dev", "test"):
        path = args.corpus / f"{name}.tsv"
        if path.is_file():
            with path.open(encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle, delimiter="\t", quoting=csv.QUOTE_NONE):
                    split_of.setdefault(row["client_id"], name)

    # Later sweeps win: verify_after re-ran the first clips against a corrected
    # lexicon, so its candidate lists are the current ones.
    rows: dict[tuple, dict] = {}
    for path in args.sweeps:
        for row in load_sweep(path):
            row.setdefault("sentence", sentences.get(row.get("clip", ""), ""))
            key = (row.get("clip"), row.get("form"), row.get("start"))
            rows[key] = row
    raw = list(rows.values())
    counts["rows_read"] = len(raw)
    print(f"{len(raw):,} rows from {len(args.sweeps)} sweeps", flush=True)

    # Recover spans the older sweeps never stored, one API call per sentence.
    need = sorted({r["sentence"] for r in raw
                   if r.get("start") is None and r.get("sentence")})
    counts["sentences_needing_spans"] = len(need)
    spans: dict[str, list[dict]] = {}
    if need:
        print(f"recovering spans for {len(need):,} sentences…", flush=True)
        with futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            for text, tokens in zip(need, pool.map(
                    lambda t: ask(args.endpoint, t), need), strict=True):
                spans[text] = tokens
        seen: collections.Counter[tuple[str, str]] = collections.Counter()
        for row in raw:
            if row.get("start") is not None:
                continue
            key = (row["sentence"], row["form"].lower())
            index = seen[key]
            seen[key] += 1
            matches = [t for t in spans.get(row["sentence"], [])
                       if t["text"].lower() == row["form"].lower()]
            if index < len(matches):
                row["start"] = matches[index].get("start")
                row["end"] = matches[index].get("end")
    counts["rows_without_span"] = sum(1 for r in raw if r.get("start") is None)

    # Step 2 — collapse speakers on (sentence, span).
    grouped: dict[tuple, list[dict]] = collections.defaultdict(list)
    for row in raw:
        if row.get("start") is None or not row.get("sentence"):
            continue
        grouped[(row["sentence"], row["start"], row["end"])].append(row)
    collapsed = []
    split_votes = 0
    for (sentence, start, end), group in grouped.items():
        picks = collections.Counter(r["audio"] for r in group)
        top, votes = picks.most_common(1)[0]
        if len(group) > 1 and votes < len(group):
            split_votes += 1
            continue
        best = max(group, key=lambda r: r["confidence"])
        collapsed.append({
            "sentence": sentence, "start": start, "end": end,
            "form": best["form"], "candidates": best.get("candidates") or [],
            "gold": str(top), "audio_confidence": best["confidence"],
            "speakers": len({speakers.get(r["clip"]) for r in group}),
            "pipeline_agreed": best.get("text") == top,
            "clip": best["clip"],
        })
    counts["after_collapse"] = len(collapsed)
    counts["dropped_speakers_split"] = split_votes
    print(f"collapsed to {len(collapsed):,} spans "
          f"({split_votes:,} dropped where speakers disagreed)", flush=True)

    # Step 3 — gate, in order, counting each drop.
    ambiguous, negatives = [], []
    drops: collections.Counter[str] = collections.Counter()
    per_form: collections.Counter[str] = collections.Counter()
    random.Random(args.seed).shuffle(collapsed)
    for row in collapsed:
        candidates = row["candidates"]
        if len(candidates) < 2:
            if len(candidates) == 1 and row["gold"] == candidates[0]:
                negatives.append({**row, "source": "lexicon-unambiguous", "weight": 1.0})
            else:
                drops["not_ambiguous"] += 1
            continue
        if row["gold"] not in candidates:
            drops["reading_not_offered"] += 1
            continue
        # A compound word can carry two accents, written "0|1". Two rows in
        # 27,040 do; a classifier over single positions has nowhere to put
        # them and they are not the homograph problem.
        if not all(c.isdigit() for c in candidates):
            drops["compound_signature"] += 1
            continue
        strong = (row["audio_confidence"] >= args.min_confidence
                  or (row["speakers"] >= 2 and row["audio_confidence"] >= args.pair_confidence))
        if not strong:
            drops["low_confidence"] += 1
            continue
        if is_clitic(row["sentence"], row["form"]):
            drops["prepositional_blind_spot"] += 1
            continue
        form = row["form"].lower()
        if per_form[form] >= args.per_form_cap:
            drops["per_form_cap"] += 1
            continue
        per_form[form] += 1
        ambiguous.append({**row, "source": "cv-audio",
                          "weight": float(min(row["speakers"], 3))})
    counts.update({f"dropped_{k}": v for k, v in drops.items()})
    counts["ambiguous_rows"] = len(ambiguous)
    counts["ambiguous_forms"] = len(per_form)
    print(f"{len(ambiguous):,} ambiguous rows over {len(per_form):,} forms; "
          f"drops {dict(drops)}", flush=True)

    # Step 5 — negatives, sampled 1:1.
    random.Random(args.seed).shuffle(negatives)
    negatives = negatives[:len(ambiguous)]
    counts["negative_rows"] = len(negatives)

    # Step 6 — split. Speaker split from Common Voice keeps the audio side
    # disjoint; a tenth of ambiguous forms is held out entirely as well.
    forms = sorted(per_form)
    random.Random(args.seed).shuffle(forms)
    unseen = set(forms[:max(1, len(forms) // 10)])
    counts["held_out_forms"] = len(unseen)
    buckets: dict[str, list[dict]] = {"train": [], "dev": [], "test": []}
    for row in ambiguous + negatives:
        where = split_of.get(speakers.get(row["clip"], ""), "train")
        if row["form"].lower() in unseen:
            where = "test"
            row["unseen_form"] = True
        buckets[where].append(row)

    for name, part in buckets.items():
        target = args.out / f"{name}.jsonl"
        target.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in part) + "\n",
                          encoding="utf-8")
        counts[f"{name}_rows"] = len(part)
        print(f"  {name:<6}{len(part):>8,} rows")

    (args.out / "manifest.json").write_text(
        json.dumps({"counts": counts, "gates": {
            "min_confidence": args.min_confidence,
            "pair_confidence": args.pair_confidence,
            "per_form_cap": args.per_form_cap}, "seed": args.seed},
            ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nmanifest -> {args.out / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
