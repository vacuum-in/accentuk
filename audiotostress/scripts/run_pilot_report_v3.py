"""One table from the v3 pilot rows: is each book usable, and how good will
its labels be? Every row carries the lexicon's reading where there is one and
the ranker's pick always, so both readings come from the same file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--books", type=Path, default=Path("artifacts/books"))
    parser.add_argument("--suffix", default="pilot3")
    parser.add_argument("--min-rows", type=int, default=50)
    parser.add_argument("--lift", type=float, default=8.0,
                        help="pp over chance the longest vowel must make; "
                             "the void run made +1 to +3, usable books +10 to +30")
    parser.add_argument("--ranker", type=float, default=0.90)
    parser.add_argument("--out", type=Path, default=Path("artifacts/books/pilot_keep_v3.json"))
    args = parser.parse_args()

    print("| book | rows | lexicon | longest vowel | chance | lift | ranker | ≥0.99 | share | verdict |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    keep, dropped = [], []
    totals = [0, 0, 0.0, 0, 0]
    for path in sorted(args.books.glob(f"*.{args.suffix}.jsonl")):
        slug = path.name[: -len(f".{args.suffix}.jsonl")]
        rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        named = [r for r in rows if r.get("label") is not None
                 and 0 <= r["label"] < len(r["durations_ms"])]
        hits = sum(max(range(len(r["durations_ms"])),
                       key=lambda i, r=r: r["durations_ms"][i]) == r["label"] for r in named)
        chance = sum(1 / len(r["durations_ms"]) for r in named)
        agree = sum(r["audio"] == r["label"] for r in named)
        strong = [r for r in named if r["confidence"] >= 0.99]
        agree_strong = sum(r["audio"] == r["label"] for r in strong)
        n = len(named)
        vowel, base = (hits / n, chance / n) if n else (0.0, 0.0)
        ranker = agree / n if n else 0.0
        lift = 100 * (vowel - base)
        ok = len(rows) >= args.min_rows and n >= 100 and lift >= args.lift and ranker >= args.ranker
        (keep if ok else dropped).append(slug)
        totals[0] += hits; totals[1] += n; totals[2] += chance; totals[3] += agree
        print(f"| {slug[:40]} | {len(rows):,} | {n:,} | {vowel:.0%} | {base:.0%} | {lift:+.1f} "
              f"| {ranker:.1%} | {agree_strong / max(len(strong), 1):.1%} "
              f"| {len(strong) / max(n, 1):.0%} | {'use' if ok else 'DROP'} |")
    if totals[1]:
        print(f"\nall: longest vowel {totals[0] / totals[1]:.1%} over {totals[1]:,} words "
              f"(chance {totals[2] / totals[1]:.1%}); ranker {totals[3] / totals[1]:.2%}")
    print(f"{len(keep)} books pass, {len(dropped)} dropped -> {args.out}")
    for slug in dropped:
        print(f"  DROP {slug}")
    args.out.write_text(json.dumps(keep, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
