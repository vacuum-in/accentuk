"""One table: is each book's audio usable, and how good will its labels be?

Two independent readings, because they fail in different ways. The longest
vowel needs no model at all and catches boundaries that are not measurements.
The ranker's agreement with the lexicon says what the labels themselves will
be worth, and its confidence says where the gate belongs.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def wilson(hits: int, total: int) -> tuple[float, float]:
    if not total:
        return 0.0, 0.0
    z, p = 1.96, hits / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    spread = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, centre - spread), min(1.0, centre + spread)


def read(path: Path):
    if not path.exists():
        return []
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--books", type=Path,
                        default=Path("/home/devops/audiotostress/artifacts/books"))
    parser.add_argument("--vowel-lift", type=float, default=0.08,
                        help="percentage points the longest-vowel baseline must "
                             "clear chance by. An absolute floor is arbitrary — "
                             "chance itself moves with how many vowels the "
                             "words have — and the discriminator is the lift: "
                             "the void mining run managed +1 to +3, every "
                             "usable book here makes +10 or better")
    parser.add_argument("--ranker-floor", type=float, default=0.90)
    args = parser.parse_args()

    print("| book | rows | longest vowel | chance | lift | ranker | ranker ≥0.99 | share | verdict |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    verdicts = {}
    totals = [0, 0, 0, 0]
    for path in sorted(args.books.glob("*.pilot.jsonl")):
        slug = path.name[: -len(".pilot.jsonl")]
        rows = read(path)
        hits = total = 0
        chance = 0.0
        for row in rows:
            durations = row.get("durations_ms") or []
            label = row.get("label")
            if label is None or len(durations) < 2 or not 0 <= label < len(durations):
                continue
            hits += max(range(len(durations)), key=lambda i: durations[i]) == label
            chance += 1 / len(durations)
            total += 1
        calib = read(args.books / f"{slug}.pilot.calib.jsonl")
        good = sum(r["audio"] == r["label"] for r in calib)
        strong = [r for r in calib if r["confidence"] >= 0.99]
        strong_good = sum(r["audio"] == r["label"] for r in strong)

        vowel = hits / total if total else 0.0
        base = chance / total if total else 0.0
        ranker = good / len(calib) if calib else 0.0
        gated = strong_good / len(strong) if strong else 0.0
        ok = (total >= 200 and vowel - base >= args.vowel_lift
              and len(calib) >= 200 and ranker >= args.ranker_floor)
        verdicts[slug] = ok
        totals[0] += hits
        totals[1] += total
        totals[2] += good
        totals[3] += len(calib)
        print(f"| {slug[:38]} | {len(rows):,} | {vowel:.0%} | {base:.0%} "
              f"| {100 * (vowel - base):+.0f} pp | {ranker:.1%} | {gated:.1%} "
              f"| {len(strong) / len(calib):.0%} " if calib else
              f"| {slug[:38]} | {len(rows):,} | — | — | — | — | — | — ",
              end="")
        print(f"| {'use' if ok else 'DROP'} |")

    if totals[1] and totals[3]:
        low, high = wilson(totals[2], totals[3])
        print(f"\nall books: longest vowel {totals[0] / totals[1]:.1%} "
              f"over {totals[1]:,} words; ranker {totals[2] / totals[3]:.2%} "
              f"[{low:.1%}, {high:.1%}] over {totals[3]:,}")
    print("Common Voice, the same two: 63.4% and 96.8% at the 0.99 gate.")
    dropped = [s for s, ok in verdicts.items() if not ok]
    if dropped:
        print(f"\n{len(dropped)} books fail the floors and must not be mined:")
        for slug in dropped:
            print(f"  {slug}")
    keep = [s for s, ok in verdicts.items() if ok]
    (args.books / "pilot_keep.json").write_text(
        json.dumps(keep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{len(keep)} books pass -> artifacts/books/pilot_keep.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
