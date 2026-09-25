"""Correct lexicon stresses the source trie disagrees with.

`run_recover_readings.py` adds a *second* reading where the lexicon holds one.
This handles the other failure: the lexicon holds one reading and it is simply
wrong. Both come from the same cause — `data/sterss-dict/run_stress.py` ran the
stressifier over a bare word list with Stanza disambiguation, so every word was
parsed in isolation as a nominative singular and Stanza chose among the trie's
readings on features that do not exist without a sentence.

Scope, measured: of 1,848,453 forms the lexicon records with one signature and
the trie records with one reading, 8,703 disagree — 0.5% by type. That number
understates the damage because the disagreements are concentrated in common
words: `украї́ни` read as `укра́їни`, `дія́льність` as `ді́яльність`, `воно́` as
`во́но`, `о́бластях` as `областя́х`.

Only forms where the trie is **unambiguous** are corrected. Where the trie
records two readings the lexicon's choice may be a legitimate sense preference,
and that is the recovery script's business, not this one's.
"""

from __future__ import annotations

import argparse
import collections
import json
import unicodedata
from pathlib import Path
from typing import Any

import psycopg
from psycopg import sql
from ukrainian_word_stress.stressify_ import (
    _load_dictionary,
    _parse_dictionary_value,
    _trie_value,
)
from ukstress.database import build_lookup_projection
from ukstress.normalizer import (
    ACUTE,
    NormalizationError,
    canonical_stressed_form,
    lookup_key,
    stress_signature,
    validate_stress,
)


def scan(conn: psycopg.Connection, dataset: int,
         overridden: frozenset[str]) -> list[dict[str, Any]]:
    signatures: dict[str, set[str]] = collections.defaultdict(set)
    with conn.cursor(name="corrections") as cursor:
        cursor.itersize = 200_000
        cursor.execute(
            "SELECT form_normalized, stress_signature FROM stress_lookup "
            "WHERE dataset_id = %s", (dataset,))
        for form, signature in cursor:
            signatures[str(form)].add(str(signature))

    trie = _load_dictionary()
    corrections: list[dict[str, Any]] = []
    for form, present in signatures.items():
        if len(present) != 1 or form in overridden:
            continue
        # `form_normalized` is NFD; the trie's keys are NFC. Looking the NFD
        # string up silently missed every form containing a decomposable
        # letter -- `ї`, `й`, and so on -- which is most of the interesting
        # ones. `україни`, `дітей` and `україні` all have the right stress in
        # the trie and none of them reached this dataset. The accent positions
        # the trie returns index the composed string too, so the stressed form
        # has to be built from it, not from `form`.
        composed = unicodedata.normalize("NFC", form)
        values = _trie_value(trie, composed)
        if values is None:
            continue
        accents = {tuple(a) for _, a in _parse_dictionary_value(values[0])}
        if len(accents) != 1:
            continue
        stressed = composed
        for position in sorted(next(iter(accents)), reverse=True):
            stressed = stressed[:position] + ACUTE + stressed[position:]
        stressed = canonical_stressed_form(stressed)
        if validate_stress(stressed).classification == "rejected":
            continue
        try:
            wanted = stress_signature(stressed)
        except NormalizationError:
            continue
        if wanted in present:
            continue
        corrections.append({"form": form, "was": next(iter(present)),
                            "signature": wanted, "stressed": stressed})
    corrections.sort(key=lambda r: r["form"])
    return corrections


def write_dataset(dsn: str, key: str, corrections: list[dict[str, Any]],
                  confidence: float = 0.99, source_rank: int = 1,
                  source_kind: str = "trie_correction") -> int:
    """Write corrections at a confidence that outranks the merged wordlist.

    0.99 sits above the wordlist's 0.95 so a correction actually takes effect,
    and below 1.00 so a human decision recorded at full confidence still wins —
    which is what `confidence` is for: a reviewed form is written at 1.00 and
    beats every generated dataset, including this one.
    """
    written = 0
    with psycopg.connect(dsn) as conn, conn.transaction():
        row = conn.execute("SELECT id FROM import_run WHERE dataset_key=%s", (key,)).fetchone()
        if row:
            dataset_id = int(row[0])
            for table in ("source_ref", "stress_variant", "word_form", "lexeme"):
                conn.execute(sql.SQL("DELETE FROM {} WHERE dataset_id = %s").format(
                    sql.Identifier(table)), (dataset_id,))
        else:
            dataset_id = int(conn.execute(
                """INSERT INTO import_run (dataset_key, status, dump_url, dump_sha256,
                       parser_version, normalization_version, schema_version, statistics)
                   VALUES (%s,'building','trie://corrections',%s,'trie-correct-1','1','007','{}')
                   RETURNING id""", (key, "0" * 64)).fetchone()[0])
        for record in corrections:
            form, stressed, signature = record["form"], record["stressed"], record["signature"]
            natural = f"{form}:{signature}"
            lexeme_id = int(conn.execute(
                """INSERT INTO lexeme (dataset_id, lemma, lemma_normalized, stressed_lemma,
                       sense_key, source_title, source_section, confidence, natural_key)
                   VALUES (%s,%s,%s,%s,'0',%s,'TrieCorrection',%s,%s) RETURNING id""",
                (dataset_id, form, form, stressed, form, confidence,
                 f"fix:{key}:{natural}")).fetchone()[0])
            word_form_id = int(conn.execute(
                """INSERT INTO word_form (dataset_id, lexeme_id, form, form_normalized,
                       morphology_key, is_lemma, is_variant, confidence, source_rank, natural_key)
                   VALUES (%s,%s,%s,%s,'',true,false,%s,%s,%s) RETURNING id""",
                (dataset_id, lexeme_id, stressed, form, confidence, source_rank,
                 f"fix:wf:{natural}")).fetchone()[0])
            conn.execute(
                """INSERT INTO stress_variant (dataset_id, word_form_id, stressed_form,
                       stress_signature, variant_type, confidence, natural_key)
                   VALUES (%s,%s,%s,%s,'primary',%s,%s)""",
                (dataset_id, word_form_id, stressed, signature, confidence,
                 f"fix:sv:{natural}"))
            conn.execute(
                """INSERT INTO source_ref (dataset_id, lexeme_id, word_form_id, page_title,
                       source_kind, source_fragment, natural_key)
                   VALUES (%s,%s,%s,%s,%s,%s,%s)""",
                (dataset_id, lexeme_id, word_form_id, form, source_kind, stressed,
                 f"fix:sr:{natural}"))
            written += 1
        build_lookup_projection(conn, dataset_id)
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database-url",
                        default="postgresql://ukstress_owner:ukstress_owner@localhost:5432/ukstress")
    parser.add_argument("--dataset", type=int, default=2)
    parser.add_argument("--dataset-key", default="trie-corrections-v1")
    parser.add_argument("--overrides", type=Path,
                        default=Path("output/ml/stress_overrides.json"))
    parser.add_argument("--out", type=Path, default=Path("output/ml/trie_corrections.json"))
    parser.add_argument("--write-db", action="store_true")
    args = parser.parse_args()

    overridden = set()
    if args.overrides.exists():
        overridden = {lookup_key(str(e["form"]))
                      for e in json.loads(args.overrides.read_text(encoding="utf-8"))
                      if e.get("form")}

    with psycopg.connect(args.database_url) as conn:
        corrections = scan(conn, args.dataset, frozenset(overridden))
    args.out.write_text(json.dumps(corrections, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"corrections: {len(corrections):,}  -> {args.out}")
    for record in corrections[:8]:
        print(f"   {record['form']:18} sig {record['was']} -> {record['signature']}  "
              f"{record['stressed']}")
    if args.write_db:
        print(f"wrote {write_dataset(args.database_url, args.dataset_key, corrections):,} rows "
              f"to '{args.dataset_key}'")


if __name__ == "__main__":
    main()
