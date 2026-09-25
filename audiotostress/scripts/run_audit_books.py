"""The cheapest check that book rows are measurements and not noise.

No GPU, no model, no labels beyond the lexicon's own. A stressed Ukrainian
vowel is longer than its neighbours, so picking the longest vowel guesses the
stress well above chance wherever the boundaries are real: 63.4% against 37.6%
chance on Common Voice. The first book mining run scored 36–39% against 35–38%,
and that is how two hundred thousand labels were found to be worthless — after
they had been built, not before.

Run this on every book before anything reads its rows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, nargs="+", required=True)
    parser.add_argument("--lift", type=float, default=8.0,
                        help="percentage points over chance a book must clear. "
                             "An absolute floor is arbitrary because chance "
                             "moves with how many vowels the words have; the "
                             "lift is what separates a measurement from noise. "
                             "The void mining run managed +1 to +3")
    args = parser.parse_args()

    print("| book | rows | lexicon | longest vowel | chance | lift |")
    print("| --- | ---: | ---: | ---: | ---: | ---: |")
    bad = []
    for path in sorted(args.rows):
        hits = total = 0
        chance = 0.0
        lines = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            lines += 1
            row = json.loads(line)
            durations = row.get("durations_ms") or []
            label = row.get("label")
            if label is None or not 0 <= label < len(durations) or len(durations) < 2:
                continue
            hits += max(range(len(durations)),
                        key=lambda i: durations[i]) == label
            chance += 1 / len(durations)
            total += 1
        if not total:
            print(f"| {path.name[:-11]} | {lines:,} | 0 | — | — | — |")
            continue
        score, base = hits / total, chance / total
        name = path.name.replace(".rows2.jsonl", "").replace(".rows.jsonl", "")
        print(f"| {name} | {lines:,} | {total:,} | {score:.1%} "
              f"| {base:.1%} | {100 * (score - base):+.1f} pp |")
        if 100 * (score - base) < args.lift:
            bad.append((name, 100 * (score - base)))

    print(f"\nCommon Voice, the same baseline: 63.4% against 37.6%.")
    if bad:
        print("\nbelow the floor, do not use:")
        for name, lift in bad:
            print(f"  {name}  {lift:+.1f} pp")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
