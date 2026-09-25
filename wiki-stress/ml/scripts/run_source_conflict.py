"""Who is right where the trie and the parsed Wiktionary disagree.

The lexicon layers sources by a constant: the merged Wiktionary wordlist sits at
0.95 or 0.90 depending on which parse produced it, and every trie-derived
dataset sits at 0.99, above all of it. That was set because the trie caught
8,702 stresses the parse got wrong. Applied globally it also loses wherever the
parse was right — `Ки́їв` is correct at 0.95 in the wordlist and was overridden
by two trie datasets at 0.99.

A constant cannot be the answer to a question that has two sides. This scores
both sides on the benchmark's own gold, split by which Wiktionary confidence
layer the disagreement is in, so the layering can be set from evidence.
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import re
import sys
import unicodedata
from pathlib import Path

import psycopg

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "etl" / "src"))
from ukstress.normalizer import NormalizationError, stress_signature

log = logging.getLogger(__name__)

WORD = re.compile(r"[А-Яа-яЇїІіЄєҐґ'’+\-]+")
TRIE_DATASETS = (4, 6, 9, 10, 11)
WIKTIONARY = 2


def gold_tokens(path: Path) -> dict[str, collections.Counter[str]]:
    """Signature per surface form, as the reference marks it.

    A form is kept only when the reference is unanimous about it: a word the
    gold stresses two ways across sentences is context-dependent, and neither
    source can be scored on it without knowing which sentence is meant.
    """
    seen: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    for row in json.loads(path.read_text(encoding="utf-8")):
        for token in WORD.findall(row["gold"]):
            if "+" not in token:
                continue
            marked = unicodedata.normalize("NFC", token.replace("+", "́"))
            form = unicodedata.normalize("NFD", marked.replace("́", "")).lower()
            try:
                seen[form][stress_signature(marked)] += 1
            except (NormalizationError, ValueError) as error:
                log.debug("skipping %r: %s", token, error)
    return seen


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path,
                        default=Path("output/ml/live_review3.json"),
                        help="a run_live_bench --save file, read for its gold column")
    parser.add_argument("--database-url",
                        default="postgresql://ukstress_owner:ukstress_owner@localhost:5432/ukstress")
    parser.add_argument("--out", type=Path,
                        default=Path("output/ml/source_conflict.json"))
    args = parser.parse_args()

    gold = gold_tokens(args.review)
    unanimous = {f: c.most_common(1)[0][0] for f, c in gold.items() if len(c) == 1}
    print(f"gold forms: {len(gold):,}   unanimous: {len(unanimous):,}", flush=True)

    forms = list(unanimous)
    with psycopg.connect(args.database_url) as conn:
        rows = conn.execute(
            """
            SELECT form_normalized, dataset_id, stress_signature, confidence, source_rank,
                   is_obsolete, is_lemma, stressed_form
            FROM stress_lookup
            WHERE form_normalized = ANY(%s) AND dataset_id = ANY(%s)
            """,
            (forms, [WIKTIONARY, *TRIE_DATASETS]),
        ).fetchall()
    print(f"lexicon rows for those forms: {len(rows):,}", flush=True)

    by_form: dict[str, list[tuple]] = collections.defaultdict(list)
    for row in rows:
        by_form[row[0]].append(row)

    # The API's own ordering, so "what Wiktionary says" here is what it serves.
    def best(candidates: list[tuple]) -> tuple | None:
        if not candidates:
            return None
        return min(candidates, key=lambda r: (r[5], -r[3], r[4], not r[6], r[7]))

    tally: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    examples: dict[str, list[str]] = collections.defaultdict(list)
    for form, want in unanimous.items():
        rows_for = by_form.get(form, [])
        wiki = best([r for r in rows_for if r[1] == WIKTIONARY])
        trie = best([r for r in rows_for if r[1] in TRIE_DATASETS])
        if wiki is None or trie is None:
            continue
        if wiki[2] == trie[2]:
            tally["agree"][f"{wiki[3]:.2f}"] += 1
            continue
        layer = f"{wiki[3]:.2f}"
        if wiki[2] == want:
            tally["wiktionary_right"][layer] += 1
            if len(examples[layer + ":wiki"]) < 5:
                examples[layer + ":wiki"].append(f"{form}: wiki {wiki[7]} vs trie {trie[7]}")
        elif trie[2] == want:
            tally["trie_right"][layer] += 1
            if len(examples[layer + ":trie"]) < 5:
                examples[layer + ":trie"].append(f"{form}: trie {trie[7]} vs wiki {wiki[7]}")
        else:
            tally["both_wrong"][layer] += 1

    print("\ndisagreements, by the Wiktionary confidence layer they sit in")
    print(f"{'layer':<8}{'wiki right':>12}{'trie right':>12}{'both wrong':>12}{'trie wins %':>13}")
    layers = sorted({k for c in tally.values() for k in c}, reverse=True)
    for layer in layers:
        w = tally["wiktionary_right"][layer]
        t = tally["trie_right"][layer]
        b = tally["both_wrong"][layer]
        share = 100.0 * t / (t + w) if (t + w) else float("nan")
        print(f"{layer:<8}{w:>12}{t:>12}{b:>12}{share:>12.1f}%")
    print(f"\nagreed outright: {sum(tally['agree'].values()):,}")

    for layer in layers:
        for side in ("wiki", "trie"):
            key = f"{layer}:{side}"
            if examples.get(key):
                print(f"\n{layer}, {side} right:")
                for line in examples[key]:
                    print("   ", line)

    args.out.write_text(json.dumps(
        {k: dict(v) for k, v in tally.items()}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
