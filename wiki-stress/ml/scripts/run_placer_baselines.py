"""Score every cheap way to guess a stress, before training anything.

`PLAN_NEXT.md` records the lesson this exists to honour: the session's biggest
gains were bug fixes and data imports, and the one retrain scored −0.3. So the
placer gets its baselines first. If a longest-suffix lookup over 1.9M lexicon
forms already reaches the target, a character transformer is not worth the
week it costs.

Two test sets, and they answer different questions:

* **target** — the 520 benchmark tokens the pipeline actually routed to the
  suffix or compound fallback, with lang-uk's gold. This is the acceptance
  set: the distribution the placer will really see, which is skewed towards
  proper nouns, compounds and loanwords.
* **held-out lexicon** — a random slice of the 1.89M single-reading forms.
  Large enough for tight intervals, but these are words the lexicon knows, so
  a lookup baseline scores far higher here than it can in production. Reported
  to show the gap, not to accept against.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import random
import unicodedata
from pathlib import Path

VOWELS = "аеєиіїоуюяй"   # the pipeline's alphabet: й counts


def vowel_positions(form: str) -> list[int]:
    return [i for i, c in enumerate(form) if c.lower() in VOWELS]


def wilson(hits: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if not total:
        return 0.0, 0.0
    p = hits / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    spread = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, centre - spread), min(1.0, centre + spread)


def build_suffix_index(pool: list[dict], max_len: int = 8) -> dict[str, int]:
    """Longest-suffix majority vote, counted from the tail so the ordinal is
    stable across word lengths: a suffix predicts *which vowel from the end*
    carries the stress, not which from the start."""
    votes: dict[str, collections.Counter[int]] = collections.defaultdict(collections.Counter)
    for row in pool:
        form, gold = row["form"], row["gold"]
        positions = vowel_positions(form)
        from_end = len(positions) - 1 - gold
        for length in range(1, min(max_len, len(form)) + 1):
            votes[form[-length:]][from_end] += 1
    return {suffix: counter.most_common(1)[0][0] for suffix, counter in votes.items()}


def predict_suffix_index(index: dict[str, int], form: str, max_len: int = 8) -> int | None:
    positions = vowel_positions(form)
    for length in range(min(max_len, len(form)), 0, -1):
        hit = index.get(form[-length:])
        if hit is not None and 0 <= len(positions) - 1 - hit < len(positions):
            return len(positions) - 1 - hit
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, default=Path("output/ml/placer/pool.jsonl"))
    parser.add_argument("--target", type=Path,
                        default=Path("output/ml/placer/target_gold.json"))
    parser.add_argument("--shipped", type=Path,
                        default=Path("output/ml/suffix_table.json"),
                        help="the SUFFIX_TABLE the API serves today")
    parser.add_argument("--holdout", type=int, default=40000)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    pool = [json.loads(line) for line in
            args.pool.read_text(encoding="utf-8").splitlines() if line.strip()]
    random.Random(args.seed).shuffle(pool)
    held, train = pool[:args.holdout], pool[args.holdout:]
    target = json.loads(args.target.read_text(encoding="utf-8"))
    print(f"pool {len(pool):,} forms | train {len(train):,} | held out {len(held):,} "
          f"| target {len(target):,} tokens")

    # The shipped table maps a suffix to a stress position; read it the way the
    # Go fallback does, as an index from the end of the word.
    shipped = json.loads(args.shipped.read_text(encoding="utf-8"))

    def by_shipped(form: str) -> int | None:
        positions = vowel_positions(form)
        for length in range(min(12, len(form)), 0, -1):
            hit = shipped.get(form[-length:])
            if hit is None:
                continue
            index = len(positions) - hit
            if 0 <= index < len(positions):
                return index
        return None

    print("\nbuilding the longest-suffix index over the training pool…", flush=True)
    index = build_suffix_index(train)
    print(f"  {len(index):,} suffixes")

    common = collections.Counter()
    for row in train:
        common[(len(vowel_positions(row["form"])), row["gold"])] += 1
    frequent: dict[int, int] = {}
    for (count, gold), n in common.items():
        best = frequent.get(count)
        if best is None or n > common[(count, best)]:
            frequent[count] = gold

    strategies = {
        "first vowel": lambda f: 0,
        "last vowel": lambda f: len(vowel_positions(f)) - 1,
        "penultimate vowel": lambda f: max(len(vowel_positions(f)) - 2, 0),
        "most frequent for length": lambda f: frequent.get(len(vowel_positions(f)), 0),
        "shipped suffix table": by_shipped,
        "longest-suffix lookup": lambda f: predict_suffix_index(index, f),
    }

    for name, rows in (("target (benchmark fallback tokens)", target),
                       ("held-out lexicon forms", held)):
        print(f"\n{name} — {len(rows):,}")
        print(f"  {'strategy':<26}{'accuracy':>10}{'95% CI':>18}{'no answer':>11}")
        for label, fn in strategies.items():
            hit = seen = blank = 0
            for row in rows:
                guess = fn(row["form"])
                if guess is None:
                    blank += 1
                    seen += 1
                    continue
                seen += 1
                hit += int(guess == row["gold"])
            low, high = wilson(hit, seen)
            print(f"  {label:<26}{100 * hit / seen:>9.1f}%"
                  f"{f'[{100*low:.1f}, {100*high:.1f}]':>18}{blank:>11,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
