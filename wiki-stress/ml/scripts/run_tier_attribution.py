"""Attribute the benchmark's remaining loss to the tier that produced it.

The official evaluator reports one number per metric. It cannot say which tier
lost the token, and every wrong priority in this project came from reading an
aggregate. This reproduces the official word accuracy exactly — verified at
95.96% against `run_live_bench.py` — and breaks it down by the `status` the API
reported for each token.

Reproducing it exactly matters more than it sounds. Two earlier attempts at
this attribution disagreed with the official figure by fifteen points, and both
times the cause was the same: **the evaluator does not count `й` as a vowel**
(`[АаЕеЄєИиІіЇїОоУуЮюЯя]`) while this pipeline does, so words like `мій` are
single-vowel there and multi-vowel here. Read through the wrong alphabet,
`not_found` looked like 15% of all tokens; under the evaluator's own rules it
is eight.

Run `run_live_bench.py --save output/ml/live_reviewN.json` first: this reads
the per-token payload that saves.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import unicodedata
from pathlib import Path

# The evaluator's alphabet, not the pipeline's.
EVALUATOR_VOWELS = re.compile(r"[АаЕеЄєИиІіЇїОоУуЮюЯя]")
TOKEN = re.compile(r"[А-Яа-яЇїІіЄєҐґ'’-]+")


def stress_positions(word: str, mark: str = "+") -> set[int]:
    """Character offsets the mark sits after, ignoring the marks themselves."""
    found, index = set(), 0
    for char in word:
        if char == mark:
            found.add(index)
        else:
            index += 1
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("review", type=Path,
                        help="the --save output of run_live_bench.py")
    args = parser.parse_args()

    rows = json.loads(args.review.read_text(encoding="utf-8"))
    tiers: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    missing: collections.Counter[str] = collections.Counter()
    mismatched = 0

    for row in rows:
        gold_words = row["gold"].split()
        got_words = row["got"].split()
        if len(gold_words) != len(got_words):
            mismatched += 1
            continue
        tokens = [t for t in row.get("tokens", []) if t.get("status")]
        cursor = 0
        for gold, got in zip(gold_words, got_words, strict=True):
            # The evaluator's own exclusions, in its order: a multi-vowel word
            # the annotators left unmarked is dropped from the denominator, and
            # so is any word with one vowel or none.
            if "+" not in gold and len(EVALUATOR_VOWELS.findall(gold)) > 1:
                continue
            if len(EVALUATOR_VOWELS.findall(gold)) <= 1:
                continue

            bare = unicodedata.normalize(
                "NFC", "".join(TOKEN.findall(gold.replace("+", ""))).lower())
            while (cursor < len(tokens) and unicodedata.normalize(
                    "NFC", tokens[cursor]["text"].lower()) != bare):
                cursor += 1
            status = tokens[cursor].get("status", "?") if cursor < len(tokens) else "?"
            if cursor < len(tokens):
                cursor += 1

            correct = (gold.replace("+", "") == got.replace("+", "")
                       and "+" in got
                       and stress_positions(got) <= stress_positions(gold))
            tiers[status][0] += int(correct)
            tiers[status][1] += 1
            if not correct and status == "not_found":
                missing[bare] += 1

    total = sum(n for _, n in tiers.values())
    right = sum(h for h, _ in tiers.values())
    print(f"scored {total:,} words, {right:,} right ({100 * right / total:.2f}%)")
    if mismatched:
        print(f"{mismatched} sentences skipped on a token-count mismatch")

    print(f"\n{'tier':<22}{'words':>8}{'accuracy':>10}{'lost':>7}{'share of loss':>15}")
    lost_total = total - right
    for status, (hit, seen) in sorted(tiers.items(), key=lambda kv: -(kv[1][1] - kv[1][0])):
        lost = seen - hit
        if seen < 10:
            continue
        print(f"  {status:<20}{seen:>8,}{100 * hit / seen:>9.1f}%{lost:>7,}"
              f"{100 * lost / max(lost_total, 1):>14.0f}%")

    if missing:
        print(f"\nunanswered words that needed an accent: {sum(missing.values())} "
              f"over {len(missing)} forms")
        print("  " + ", ".join(list(missing)[:12]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
