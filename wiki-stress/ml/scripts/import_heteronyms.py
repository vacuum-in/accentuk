"""Import lang-uk's curated heteronym dictionary.

https://github.com/lang-uk/ukrainian-heteronyms-dictionary

37,030 headwords, each listing every stressed reading, maintained by hand. This
is a better authority than the two mechanisms this project used to reach the
same conclusion:

* `run_recover_readings.py` infers a second reading from the source trie's tag
  structure. That works where the trie disagrees with itself and not otherwise.
* `run_homograph_audit.py` asks two models whether a form is a heteronym. It is
  conservative — 19 agreements from 1,200 candidates — and needs API quota.

Measured against the current lexicon: 81.6% of the dictionary's groups are
already fully covered, 11.8% are missing at least one reading, and 6.2% are
absent altogether. That is 6,828 forms whose ambiguity the pipeline cannot see,
so it serves them from a single reading without ever consulting the model.

Readings land at confidence 0.93 — below the Wiktionary parser's 0.95 so a
curated entry still wins on ordering, and above the trie recovery's 0.60 and the
LLM audit's 0.55, which are weaker evidence for the same claim.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any

import psycopg
from psycopg import sql
from ukstress.database import build_lookup_projection
from ukstress.normalizer import (
    NormalizationError,
    canonical_stressed_form,
    lookup_key,
    stress_signature,
    validate_stress,
)


def read_dictionary(path: Path) -> dict[str, dict[str, str]]:
    """Return {form: {signature: stressed}} from the TSV."""
    entries: dict[str, dict[str, str]] = collections.defaultdict(dict)
    for line in path.read_text(encoding="utf-8").splitlines():
        if "\t" not in line:
            continue
        headword, readings = line.split("\t", 1)
        key = lookup_key(headword)
        for reading in readings.split(","):
            stressed = canonical_stressed_form(reading.strip())
            if not stressed or validate_stress(stressed).classification == "rejected":
                continue
            try:
                signature = stress_signature(stressed)
            except NormalizationError:
                continue
            if "|" in signature:
                # Two accents in one token is free variation, not a heteronym
                # reading a model could be asked to choose between.
                continue
            entries[key].setdefault(signature, stressed)
    return {form: readings for form, readings in entries.items() if len(readings) > 1}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsv", type=Path, required=True)
    parser.add_argument("--database-url",
                        default="postgresql://ukstress_owner:ukstress_owner@localhost:5432/ukstress")
    parser.add_argument("--dataset", type=int, action="append", default=None)
    parser.add_argument("--dataset-key", default="heteronyms-langu-v1")
    parser.add_argument("--overrides", type=Path,
                        default=Path("output/ml/stress_overrides.json"))
    parser.add_argument("--out", type=Path,
                        default=Path("output/ml/heteronym_import.json"))
    parser.add_argument("--write-db", action="store_true")
    args = parser.parse_args()

    overridden: set[str] = set()
    if args.overrides.exists():
        overridden = {lookup_key(str(e["form"]))
                      for e in json.loads(args.overrides.read_text(encoding="utf-8"))
                      if e.get("form")}

    dictionary = read_dictionary(args.tsv)
    datasets = args.dataset
    with psycopg.connect(args.database_url) as conn:
        if datasets is None:
            active = int(conn.execute(
                "SELECT dataset_id FROM active_dataset WHERE singleton").fetchone()[0])
            datasets = [active, 3, 4, 5, 6]
        have: dict[str, set[str]] = collections.defaultdict(set)
        keys = sorted(dictionary)
        for start in range(0, len(keys), 20_000):
            for form, signature in conn.execute(
                "SELECT form_normalized, stress_signature FROM stress_lookup "
                "WHERE dataset_id = ANY(%s) AND form_normalized = ANY(%s)",
                (datasets, keys[start:start + 20_000])).fetchall():
                have[str(form)].add(str(signature))

    additions: list[dict[str, Any]] = []
    tally: collections.Counter[str] = collections.Counter()
    for form, readings in dictionary.items():
        if form in overridden:
            tally["human_override"] += 1
            continue
        present = have.get(form, set())
        missing = {s: v for s, v in readings.items() if s not in present}
        if not missing:
            tally["already_complete"] += 1
            continue
        tally["absent_entirely" if not present else "partially_present"] += 1
        additions.append({"form": form, "have": sorted(present),
                          "add": [{"signature": s, "stressed": v}
                                  for s, v in sorted(missing.items())]})

    additions.sort(key=lambda r: r["form"])
    args.out.write_text(json.dumps(additions, ensure_ascii=False, indent=1), encoding="utf-8")
    for key, value in tally.most_common():
        print(f"  {value:>7,}  {key}")
    print(f"\nforms to add readings for: {len(additions):,}  -> {args.out}")

    if args.write_db:
        print(f"wrote {write_dataset(args.database_url, args.dataset_key, additions):,} rows")


def write_dataset(dsn: str, key: str, additions: list[dict[str, Any]]) -> int:
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
                   VALUES (%s,'building','https://github.com/lang-uk/ukrainian-heteronyms-dictionary',
                           %s,'heteronyms-1','1','007','{}') RETURNING id""",
                (key, "0" * 64)).fetchone()[0])
        for record in additions:
            form = record["form"]
            for item in record["add"]:
                stressed, signature = item["stressed"], item["signature"]
                natural = f"{form}:{signature}"
                lexeme_id = int(conn.execute(
                    """INSERT INTO lexeme (dataset_id, lemma, lemma_normalized, stressed_lemma,
                           sense_key, source_title, source_section, confidence, natural_key)
                       VALUES (%s,%s,%s,%s,%s,%s,'HeteronymDict',0.93,%s) RETURNING id""",
                    (dataset_id, form, form, stressed, signature, form,
                     f"het:{key}:{natural}")).fetchone()[0])
                word_form_id = int(conn.execute(
                    """INSERT INTO word_form (dataset_id, lexeme_id, form, form_normalized,
                           morphology_key, is_lemma, is_variant, confidence, source_rank,
                           natural_key)
                       VALUES (%s,%s,%s,%s,'',true,true,0.93,3,%s) RETURNING id""",
                    (dataset_id, lexeme_id, stressed, form, f"het:wf:{natural}")).fetchone()[0])
                conn.execute(
                    """INSERT INTO stress_variant (dataset_id, word_form_id, stressed_form,
                           stress_signature, variant_type, confidence, natural_key)
                       VALUES (%s,%s,%s,%s,'primary',0.93,%s)""",
                    (dataset_id, word_form_id, stressed, signature, f"het:sv:{natural}"))
                conn.execute(
                    """INSERT INTO source_ref (dataset_id, lexeme_id, word_form_id, page_title,
                           source_kind, source_fragment, natural_key)
                       VALUES (%s,%s,%s,%s,'heteronym_dictionary',%s,%s)""",
                    (dataset_id, lexeme_id, word_form_id, form, stressed, f"het:sr:{natural}"))
                written += 1
        build_lookup_projection(conn, dataset_id)
    return written


if __name__ == "__main__":
    main()
