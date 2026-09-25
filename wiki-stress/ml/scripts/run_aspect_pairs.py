"""Verb forms the trie spells one way and stresses two, with identical tags.

`засипа́ти` and `заси́пати` are the same spelling, both `VerbForm=Inf|upos=VERB`,
and the difference between them is aspect — imperfective against perfective.
The trie does not record aspect, so `separating_features` is empty and the
morphology tier declines the form outright; the model then guesses from the
sentence, and on this class it guesses badly (margins of 0.97 and 1.29 where it
is wrong, against 6 to 7 where it is right).

Aspect is not lexical trivia here: «почала», «став», «продовжували», «не можна»
all require the imperfective, which is a rule as hard as the prepositional
stress shift already in the pipeline. What is missing is one tag, not a model.

This counts the class so the cost of supplying that tag by hand can be judged
before anyone starts.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import unicodedata
from pathlib import Path
from typing import Any

import marisa_trie

TRIE_PATH = Path(
    "/home/devops/.cache/uv/archive-v0/Fd1M0Xx2Ca_isCM2/"
    "ukrainian_word_stress/data/stress.trie"
)
TAGS_PATH = "/home/devops/.cache/uv/archive-v0/Fd1M0Xx2Ca_isCM2"


def marked(form: str, accents: tuple[int, ...]) -> str:
    out = form
    for position in sorted(accents, reverse=True):
        if 0 <= position <= len(out):
            out = out[:position] + "́" + out[position:]
    return unicodedata.normalize("NFC", out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trie", type=Path, default=TRIE_PATH)
    parser.add_argument("--out", type=Path,
                        default=Path("output/ml/aspect_pairs.json"))
    parser.add_argument("--frequency", type=Path,
                        default=Path("output/ml/ambiguous_frequency.json"))
    args = parser.parse_args()

    sys.path.insert(0, TAGS_PATH)
    from ukrainian_word_stress.stressify_ import _parse_dictionary_value as parse_value

    frequency: dict[str, int] = {}
    if args.frequency.is_file():
        frequency = {str(k): int(v) for k, v
                     in json.loads(args.frequency.read_text(encoding="utf-8")).items()}

    trie = marisa_trie.BytesTrie()
    trie.load(str(args.trie))

    found: list[dict[str, Any]] = []
    shapes: collections.Counter[str] = collections.Counter()
    scanned = 0
    for form, value in trie.iteritems():
        scanned += 1
        if scanned % 500_000 == 0:
            print(f"  {scanned:,} scanned, {len(found):,} pairs", flush=True)
        readings = [(tuple(accents), frozenset(tags))
                    for tags, accents in parse_value(value)]
        verbs = [(a, t) for a, t in readings if "upos=VERB" in t]
        if len(verbs) < 2:
            continue
        # Group by the tag set: a pair that differs only in accent, with tags
        # that cannot tell the two apart, is exactly what defeats the tier.
        by_tags: dict[frozenset, set[tuple[int, ...]]] = collections.defaultdict(set)
        for accents, tags in verbs:
            by_tags[tags].add(accents)
        for tags, accents in by_tags.items():
            if len(accents) < 2:
                continue
            ordered = sorted(accents)
            shape = "|".join(sorted(t for t in tags if t != "upos=VERB")) or "(no tags)"
            shapes[shape] += 1
            found.append({
                "form": form,
                "tags": shape,
                "readings": [marked(form, a) for a in ordered],
                "frequency": frequency.get(form, 0),
            })

    found.sort(key=lambda row: (-row["frequency"], row["form"]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(found, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"\nscanned {scanned:,} trie forms")
    print(f"verb forms with two accents under one tag set: {len(found):,}")
    print("\nby tag set:")
    for shape, count in shapes.most_common(12):
        print(f"  {count:>6}  {shape}")
    seen = sum(1 for row in found if row["frequency"])
    print(f"\noccurring in the frequency sample: {seen}")
    print(f"{'freq':>7}  form / readings")
    for row in found[:20]:
        print(f"{row['frequency']:>7}  {row['form']:<16} {' / '.join(row['readings'])}")
    print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
