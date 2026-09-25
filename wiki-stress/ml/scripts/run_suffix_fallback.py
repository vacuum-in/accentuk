"""Predict stress for words the lexicon does not have, by suffix analogy.

188 benchmark tokens sit in the `uncovered` tier and score **0%**: with no
lexicon entry the pipeline emits the word unstressed, and the benchmark counts an
unstressed multi-vowel word as wrong. 176 of those 183 forms are absent from the
source trie too, so there is no dictionary to fall back to -- the stress has to
be guessed from the word's own shape.

Ukrainian stress is largely determined by the ending, so the guess is: find the
longest suffix this word shares with known lexicon forms and take their most
common stress position **counted from the end**. Counting from the end is what
makes the analogy transfer between words of different length.

This trains and evaluates the table on the lexicon itself, holding out a random
slice, so the accuracy printed is on forms the table never saw.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import unicodedata
from pathlib import Path

import psycopg

VOWELS = frozenset("аеєиіїоуюя")
BREVE = "̆"
ACUTE = "́"
# The wheel ships the trie inside the installed package.
TRIE_PATH = Path(
    "/home/devops/.cache/uv/archive-v0/Fd1M0Xx2Ca_isCM2/"
    "ukrainian_word_stress/data/stress.trie"
)


def vowels_of(word: str) -> list[int]:
    """Indices of vowel characters in the NFD string. `й` counts, as the lexicon does."""
    decomposed = unicodedata.normalize("NFD", word).lower()
    return [i for i, c in enumerate(decomposed) if c in VOWELS]


def stress_from_end(form: str, signature: str) -> int | None:
    """Stress position counted back from the last vowel: 0 = final syllable."""
    if not signature.isdigit():
        return None
    total = len(vowels_of(form))
    ordinal = int(signature)
    if total < 2 or ordinal >= total:
        return None
    return total - 1 - ordinal


def build(rows: list[tuple[str, str]], max_suffix: int) -> dict[str, collections.Counter]:
    table: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for form, signature in rows:
        position = stress_from_end(form, signature)
        if position is None:
            continue
        decomposed = unicodedata.normalize("NFD", form).lower()
        for length in range(1, min(max_suffix, len(decomposed)) + 1):
            table[decomposed[-length:]][position] += 1
    return table


def predict(word: str, table: dict[str, collections.Counter], max_suffix: int,
            min_support: int) -> int | None:
    """Longest suffix with enough evidence wins; returns the position from the end.

    The table is keyed on distance from the last vowel, and so is what this
    returns -- mixing the two conventions is what made a first run of this
    script report 22% when the table was in fact doing much better.
    """
    decomposed = unicodedata.normalize("NFD", word).lower()
    total = len(vowels_of(word))
    if total < 2:
        return None
    for length in range(min(max_suffix, len(decomposed)), 0, -1):
        counts = table.get(decomposed[-length:])
        if not counts or sum(counts.values()) < min_support:
            continue
        position, _ = counts.most_common(1)[0]
        if position < total:
            return position
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database-url",
                        default="postgresql://ukstress_owner:ukstress_owner@localhost:5432/ukstress")
    parser.add_argument("--limit", type=int, default=400000)
    parser.add_argument("--holdout", type=float, default=0.1)
    parser.add_argument("--max-suffix", type=int, default=7)
    parser.add_argument("--min-support", type=int, default=8)
    parser.add_argument("--min-purity", type=float, default=0.6,
                        help="drop a suffix whose majority position is this weak; "
                             "even a coin flip beats an unstressed word, so the "
                             "floor is about not overriding better tiers")
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--source", choices=("lexicon", "trie"), default="lexicon",
                        help="trie carries 2.9M forms against the lexicon's 400k, "
                             "which is more evidence per suffix")
    parser.add_argument("--prune", action="store_true",
                        help="drop entries a shorter suffix already answers the same way")
    parser.add_argument("--out", type=Path, default=Path("output/ml/suffix_table.json"))
    args = parser.parse_args()

    if args.source == "trie":
        import marisa_trie
        loaded = marisa_trie.BytesTrie()
        loaded.load(str(TRIE_PATH))
        rows = []
        for word in loaded.iterkeys():
            records = loaded[word]
            # One unambiguous accent only: a form the trie itself considers
            # ambiguous teaches the table nothing about where stress lands.
            positions = {r[0] for r in records if r}
            if len(positions) != 1:
                continue
            index = next(iter(positions))
            if index < 1 or index > len(word):
                continue
            prefix = unicodedata.normalize("NFD", word[:index]).lower()
            ordinal = sum(1 for c in prefix if c in VOWELS) - 1
            if ordinal < 0:
                continue
            rows.append((word, str(ordinal)))
            if len(rows) >= args.limit:
                break
    else:
        with psycopg.connect(args.database_url) as conn:
            rows = conn.execute(
                """SELECT form_normalized, min(stress_signature)
                   FROM stress_lookup
                   WHERE dataset_id = 2 AND stress_signature ~ '^[0-9]+$'
                   GROUP BY form_normalized
                   HAVING count(DISTINCT stress_signature) = 1
                   LIMIT %s""", (args.limit,)).fetchall()
        rows = [(str(f), str(s)) for f, s in rows]
    random.Random(args.seed).shuffle(rows)
    cut = int(len(rows) * args.holdout)
    held, train = rows[:cut], rows[cut:]
    print(f"lexicon forms: {len(rows):,}  train {len(train):,}  held-out {len(held):,}", flush=True)

    table = build(train, args.max_suffix)
    print(f"suffix entries: {len(table):,}", flush=True)

    answered = correct = 0
    for form, signature in held:
        truth = stress_from_end(form, signature)
        if truth is None:
            continue
        guess = predict(form, table, args.max_suffix, args.min_support)
        if guess is None:
            continue
        answered += 1
        correct += int(guess == truth)
    scored = sum(1 for f, s in held if stress_from_end(f, s) is not None)
    print(f"\nheld-out forms scored : {scored:,}")
    print(f"  answered            : {answered:,}  ({answered / scored:.1%} coverage)")
    print(f"  correct             : {correct:,}  ({correct / max(1, answered):.1%} precision)")

    # Persist only suffixes the majority position actually dominates; a coin-flip
    # suffix is worse than leaving the word alone once it reaches serving.
    compact = {suffix: counts.most_common(1)[0][0]
               for suffix, counts in table.items()
               if sum(counts.values()) >= args.min_support
               and counts.most_common(1)[0][1] / sum(counts.values()) >= args.min_purity}
    # Lookup walks from the longest suffix down, so an entry whose next-shorter
    # suffix already gives the same answer is never consulted for a different
    # result. Dropping those is lossless and takes the trie-built table from
    # 1.67M entries to a size a server can hold without thinking about it.
    if args.prune:
        before = len(compact)
        compact = {suffix: position for suffix, position in compact.items()
                   if len(suffix) == 1 or compact.get(suffix[1:]) != position}
        print(f"pruned {before - len(compact):,} redundant suffixes", flush=True)
    args.out.write_text(json.dumps(compact, ensure_ascii=False), encoding="utf-8")
    print(f"\nkept {len(compact):,} suffixes -> {args.out}")


if __name__ == "__main__":
    main()
