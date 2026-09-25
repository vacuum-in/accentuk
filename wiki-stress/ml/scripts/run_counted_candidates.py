"""Every noun whose plural and genitive singular are spelt alike but stressed apart.

That homography is what makes the counted form possible: after 2, 3 or 4 the
noun takes a form spelt like the nominative plural but stressed like the
genitive singular — `три сестри́` against `се́стри`. Where the two stresses
coincide there is nothing to choose and no candidate.

The class is derived from the stress trie rather than scraped, because the trie
is compiled from Словники України and carries the case and number tags on each
reading: a form is a candidate exactly when the same spelling appears with
`Case=Gen|Number=Sing` and with `Number=Plur` at a different accent.

Membership in the class is *not* the same as taking the counted form — that is
lexical, and no rule derives it. This produces the review queue; the verdict
comes from a person. The earlier attempt to settle it automatically, by asking
whether an orthoepic dictionary printed a numeral phrase, treated silence as a
negative and filed `вікна` as nominative, which a native speaker rejects.
"""

from __future__ import annotations

import argparse
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


def readings(value: bytes, parse: Any) -> list[tuple[tuple[int, ...], set[str]]]:
    """(accent positions, tag set) for each reading the trie records.

    Decoded by the package's own parser rather than by splitting on the record
    separator: the tag table has bytes that are not tags, and hand-decoding
    raised KeyError on the first of them.
    """
    return [(tuple(accents), set(tags)) for tags, accents in parse(value)]


def marked(form: str, accent: int) -> str:
    """The trie's accent is a 1-based character position into the composed form."""
    return unicodedata.normalize(
        "NFC", form[:accent] + "́" + form[accent:]) if accent else form


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trie", type=Path, default=TRIE_PATH)
    parser.add_argument("--out", type=Path,
                        default=Path("output/ml/counted_candidates.json"))
    parser.add_argument("--proper-names", type=Path,
                        default=Path("output/ml/propernames_v25.json"),
                        help="excluded: a name's stress is not a counted form")
    args = parser.parse_args()

    sys.path.insert(0, TAGS_PATH)
    from ukrainian_word_stress.stressify_ import _parse_dictionary_value as parse_value

    names: set[str] = set()
    if args.proper_names.is_file():
        names = {k.lower() for k in
                 json.loads(args.proper_names.read_text(encoding="utf-8"))["forms"]}

    trie = marisa_trie.BytesTrie()
    trie.load(str(args.trie))

    found = []
    scanned = 0
    for form, value in trie.iteritems():
        scanned += 1
        if scanned % 500_000 == 0:
            print(f"  {scanned:,} forms scanned, {len(found):,} candidates", flush=True)
        if form.lower() in names or not form[:1].islower():
            continue
        parsed = readings(value, parse_value)
        if len(parsed) < 2:
            continue
        # Pair the two readings within one gender. Two lexemes often share a
        # spelling — `коли` is a masculine plural at one accent and a feminine
        # genitive singular at another — and pairing across them invents a
        # counted form for a word that has none.
        pair = None
        for gender in ("Gender=Masc", "Gender=Fem", "Gender=Neut"):
            same = [(a, tags) for a, tags in parsed if gender in tags]
            gen_sg = {a for a, tags in same
                      if {"Case=Gen", "Number=Sing", "upos=NOUN"} <= tags}
            plural = {a for a, tags in same
                      if {"Number=Plur", "Case=Nom", "upos=NOUN"} <= tags}
            # One accent each: a reading with two is the dictionary's "either
            # stress is acceptable", which is not a case distinction.
            if len(gen_sg) != 1 or len(plural) != 1 or gen_sg == plural:
                continue
            (genitive,), (nominative,) = gen_sg, plural
            if len(genitive) != 1 or len(nominative) != 1:
                continue
            pair = (gender, nominative[0], genitive[0])
            break
        if pair is None:
            continue
        gender, nominative, genitive = pair
        found.append({
            "form": form,
            "gender": gender.split("=")[1],
            "nominative_plural": marked(form, nominative),
            "genitive_singular": marked(form, genitive),
            "verdict": "unreviewed",
        })

    found.sort(key=lambda row: row["form"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(found, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nscanned {scanned:,} trie forms")
    print(f"candidates: {len(found):,}  ->  {args.out}")
    for row in found[:8]:
        print(f"   {row['form']:<14} {row['nominative_plural']:<14} {row['genitive_singular']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
