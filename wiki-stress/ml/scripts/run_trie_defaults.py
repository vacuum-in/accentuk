"""Make the trie's unambiguous reading the default for an ambiguous form.

When the lexicon holds several readings for a form and nothing else decides —
the form is outside the serving manifest and the parser could not resolve it —
the pipeline serves the highest-confidence entry. That order is arbitrary with
respect to which reading is actually right.

The source trie names exactly one reading for 53,995 of the 61,474 ambiguous
forms, and measured on lang-uk's benchmark it is the better default: on the
tokens where the two disagree the trie is right 26 times against the lexicon
order's 12. Preferring it lifted unambiguous-word accuracy from 98.21% to 98.53%
and sentence accuracy from 63.94% to 65.11%.

This is written as a dataset rather than as serving logic on purpose: the rows
carry a confidence above the merged wordlist's, so the ordering the API already
does puts them first, and no code in the request path has to know about the
trie. The signature set a form offers is unchanged, so the model tier still sees
exactly the same candidates.
"""

from __future__ import annotations

import argparse
import collections

# Reuse the writer so both trie-derived datasets are built identically.
import importlib.util as _importlib_util
import json
import unicodedata
from pathlib import Path
from typing import Any

import marisa_trie
import psycopg
from ukstress.normalizer import ACUTE, NormalizationError, canonical_stressed_form

_spec = _importlib_util.spec_from_file_location(
    "run_trie_corrections", Path(__file__).with_name("run_trie_corrections.py"))
_corrections = _importlib_util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_corrections)

VOWELS = frozenset("аеєиіїоуюя")
TRIE_PATH = Path(
    "/home/devops/.cache/uv/archive-v0/Fd1M0Xx2Ca_isCM2/"
    "ukrainian_word_stress/data/stress.trie"
)


def trie_reading(trie: marisa_trie.BytesTrie, key: str) -> tuple[int, str] | None:
    """The single vowel ordinal the trie stresses, with the composed form.

    Returns None when the form is absent or the trie itself records more than
    one reading — an undecided trie has nothing to add to an undecided lexicon.
    The accent index the trie stores is a 1-based character position into the
    **composed** form, and the database's keys are NFD, so both conversions
    happen here.
    """
    composed = unicodedata.normalize("NFC", key)
    if composed not in trie:
        return None
    ordinals = set()
    for record in trie[composed]:
        if not record or not 1 <= record[0] <= len(composed):
            continue
        prefix = unicodedata.normalize("NFD", composed[: record[0]]).lower()
        ordinals.add(sum(1 for c in prefix if c in VOWELS) - 1)
    if len(ordinals) != 1:
        return None
    ordinal = next(iter(ordinals))
    return (ordinal, composed) if ordinal >= 0 else None


def stressed_at(composed: str, ordinal: int) -> str | None:
    """Place the acute after the given vowel, counting as the signatures do."""
    decomposed = unicodedata.normalize("NFD", composed)
    seen = -1
    for index, char in enumerate(decomposed):
        if char.lower() in VOWELS:
            seen += 1
            if seen == ordinal:
                # The acute goes after the letter *and* after the marks that
                # belong to it. `ї` decomposes to `і` plus a diaeresis, so
                # inserting straight after the base letter puts the acute
                # between the two and produces `киі́̈в` — 345 rows of the
                # published map were written that way.
                end = index + 1
                while end < len(decomposed) and unicodedata.combining(decomposed[end]):
                    end += 1
                marked = decomposed[:end] + ACUTE + decomposed[end:]
                try:
                    return canonical_stressed_form(marked)
                except NormalizationError:
                    return None
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exclude", type=Path,
                        help="manual corrections; forms named there are left "
                             "out, so a generated default never overrides a "
                             "reviewed reading")
    parser.add_argument("--database-url",
                        default="postgresql://ukstress_owner:ukstress_owner@localhost:5432/ukstress")
    parser.add_argument("--datasets", default="2,3,5,7,9,10",
                        help="datasets whose ambiguity this resolves")
    parser.add_argument("--dataset-key", default="trie-defaults-v1")
    parser.add_argument("--trie", type=Path, default=TRIE_PATH)
    parser.add_argument("--out", type=Path, default=Path("output/ml/trie_defaults.json"))
    parser.add_argument("--map", type=Path,
                        default=Path("output/ml/trie_defaults_map.json"),
                        help="form -> signature, the shape the API loads")
    parser.add_argument("--write-db", action="store_true")
    args = parser.parse_args()

    trie = marisa_trie.BytesTrie()
    trie.load(str(args.trie))
    datasets = [int(d) for d in args.datasets.split(",") if d.strip()]

    reviewed: set[str] = set()
    if args.exclude and args.exclude.is_file():
        reviewed = {unicodedata.normalize("NFD", str(r["form"]))
                    for r in json.loads(args.exclude.read_text(encoding="utf-8"))}
        print(f"reviewed forms held out: {len(reviewed)}")

    rows: list[dict[str, Any]] = []
    tally: collections.Counter[str] = collections.Counter()
    with psycopg.connect(args.database_url) as conn, conn.cursor(name="ambiguous") as cursor:
        cursor.itersize = 200_000
        cursor.execute(
            "SELECT form_normalized, array_agg(DISTINCT stress_signature) "
            "FROM stress_lookup WHERE dataset_id = ANY(%s) "
            "GROUP BY form_normalized HAVING count(DISTINCT stress_signature) > 1",
            (datasets,))
        for form, signatures in cursor:
            form = str(form)
            tally["ambiguous_forms"] += 1
            if form in reviewed:
                # A generated default must never contradict a reading someone
                # checked: this map only reorders, and reordering is exactly
                # how it would win.
                tally["reviewed_elsewhere"] += 1
                continue
            reading = trie_reading(trie, form)
            if reading is None:
                tally["trie_absent_or_undecided"] += 1
                continue
            ordinal, composed = reading
            if str(ordinal) not in {str(s) for s in signatures}:
                # The trie names a reading the lexicon does not offer. Adding it
                # would widen the candidate set and desynchronise the manifest.
                tally["trie_reading_not_offered"] += 1
                continue
            stressed = stressed_at(composed, ordinal)
            if stressed is None:
                tally["unrenderable"] += 1
                continue
            rows.append({"form": form, "signature": str(ordinal), "stressed": stressed})
            tally["written"] += 1
    rows.sort(key=lambda r: r["form"])

    args.out.write_text(json.dumps({"tally": dict(tally), "defaults": rows},
                                   ensure_ascii=False), encoding="utf-8")
    for name, count in tally.most_common():
        print(f"{name:28}{count:>9,}")
    args.map.write_text(json.dumps({r["form"]: r["signature"] for r in rows},
                                   ensure_ascii=False), encoding="utf-8")
    print(f"\n-> {args.out}\n-> {args.map}")

    if args.write_db:
        written = _corrections.write_dataset(args.database_url, args.dataset_key, rows)
        print(f"wrote {written:,} rows to {args.dataset_key!r}")
    else:
        print("--write-db not set; nothing written to PostgreSQL")


if __name__ == "__main__":
    main()
