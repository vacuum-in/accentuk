"""Recover second stress readings the merge dropped.

The gold evaluation attributed 112 of 347 errors to the `dictionary` tier —
forms the lexicon reports as unambiguous and therefore serves without ever
asking the model. Accuracy there was exactly 0.5000, which is what serving one
fixed reading scores on a sense-balanced test set. The words are genuine
homographs; the merge kept whichever reading its highest-confidence source
recorded and silently discarded the other.

The source trie shipped with `ukrainian-word-stress` still holds both. It is
queried directly rather than through `Stressifier`, because the Stressifier
runs Stanza first and returns *one* disambiguated reading — the same collapse
that produced the defect. `_parse_dictionary_value` returns every
(tags, accents) pair the dictionary records, and a form whose accent tuples
disagree is a form with more than one reading.

Recovered readings go into their own dataset at a confidence below the
Wiktionary parser's, so `stress_lookup` ordering can never let a trie-derived
stress outrank a curated one, and the whole batch can be rolled back by
unpublishing a single dataset.

Recovered readings are triaged before they are written. Making a form ambiguous
is not free: the Go API answers an ambiguous form the manifest does not cover
with the *unstressed* token, so a fabricated ambiguity turns a correct
dictionary hit into no answer at all. `free_variation` (one reading, two
acceptable accents) and `artifact` (three or more accents under one tag set)
are exactly the fabricated kind, so they are dropped here rather than filtered
downstream.

Two populations are recovered, not one. `--include-absent` adds forms the
lexicon does not hold *at all* but the trie does: the recovery scan below only
inspects forms already present with a single signature, so a homograph missing
from the lexicon entirely is invisible to it. `очник` is exactly that case —
absent from `stress_lookup`, two readings in the trie, and 32 gold rows scoring
0.0000 because the pipeline has nothing to serve.

Forms named in `stress_overrides.json` are skipped outright. Those are stresses
a person decided, usually by removing a reading the pipeline should stop
offering — `речення` is one — and a mechanical scan that reads the second
reading straight back out of the trie would silently undo the decision. It did,
on the first run.

What this does *not* do: decide which reading is right. That is the contextual
model's job, and the point of the exercise is to give it the chance — a form
with two signatures routes to the model, a form with one never does.
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

from ukstress_ml import triage

#: Triage classes that describe a real choice between senses. Anything else is
#: an artefact of how the source wordlist was stressed, and promoting it to an
#: ambiguity costs a working dictionary answer.
KEEP_CLASSES = frozenset({triage.HOMOGRAPH, triage.GRAMMATICAL})

SEGMENT_BOUNDARY = frozenset("- ʼ")


def trie_readings(trie: Any, form: str) -> list[tuple[int, ...]] | None:
    """Return the distinct accent tuples the trie records for `form`.

    None means the form is absent from the trie, which is different from the
    trie recording exactly one reading: the first is no evidence either way,
    the second is positive evidence that the lexicon's single entry is right.
    """
    # `form_normalized` comes out of the database in NFD; the trie's keys are
    # NFC. Passing the decomposed string silently missed every form containing
    # `ї`, `й` or any other composable letter -- the same defect that kept
    # `украї́ни` and `діте́й` out of the corrections dataset. The accent
    # positions the trie returns index the composed string, so callers must
    # place accents into it, not into `form`.
    form = unicodedata.normalize("NFC", form)
    values = _trie_value(trie, form)
    if values is None:
        return None
    return sorted({tuple(accents) for _, accents in _parse_dictionary_value(values[0])})


def apply_accents(form: str, positions: tuple[int, ...]) -> str:
    for position in sorted(positions, reverse=True):
        form = form[:position] + ACUTE + form[position:]
    return canonical_stressed_form(form)


def segment_count(form: str) -> int:
    return len([part for part in form.replace("ʼ", " ").replace("-", " ").split() if part])


def read_form_list(path: Path) -> list[str]:
    """Read forms from the corpus scan's TSV (count\tform) or a JSON list."""
    if path.suffix == ".tsv":
        forms = []
        for line in path.read_text(encoding="utf-8").splitlines():
            count, _, form = line.partition("\t")
            if count.isdigit() and form:
                forms.append(lookup_key(form))
        return forms
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload if isinstance(payload, list) else payload.get("forms", [])
    return [lookup_key(r if isinstance(r, str) else r["form"]) for r in rows]


def load_overridden(path: Path) -> set[str]:
    """Forms whose stress a person has already decided; never reopen them."""
    if not path.exists():
        return set()
    return {lookup_key(str(entry["form"]))
            for entry in json.loads(path.read_text(encoding="utf-8"))
            if entry.get("form")}


def scan_absent(conn: psycopg.Connection, datasets: list[int], forms: list[str],
                overridden: frozenset[str] = frozenset()) -> list[dict[str, Any]]:
    """Recover forms the lexicon lacks entirely but the trie records ambiguously.

    The lexicon's own gap, not a collapsed reading: `stress_lookup` returns
    nothing, so the API answers `not_found` and the token is served unstressed.
    Where the trie holds two readings for such a form, both are added and the
    form becomes a normal model-tier ambiguity.
    """
    known = {str(row[0]) for row in conn.execute(
        "SELECT DISTINCT form_normalized FROM stress_lookup "
        "WHERE dataset_id = ANY(%s) AND form_normalized = ANY(%s)",
        (datasets, forms)).fetchall()}
    trie = _load_dictionary()
    recovered: list[dict[str, Any]] = []
    for form in forms:
        if form in known or form in overridden:
            continue
        readings = trie_readings(trie, form)
        if readings is None or len(readings) < 2:
            continue
        expected = segment_count(form)
        usable = [r for r in readings if len(r) == expected]
        if len(usable) < 2:
            continue
        variants: dict[str, str] = {}
        for accents in usable:
            candidate = apply_accents(form, accents)
            if validate_stress(candidate).classification == "rejected":
                continue
            try:
                variants.setdefault(stress_signature(candidate), candidate)
            except NormalizationError:
                continue
        if len(variants) < 2:
            continue
        recovered.append({
            "form": form,
            "existing_signature": None,
            "existing_stressed": None,
            "triage_class": "absent_from_lexicon",
            "needs_model": True,
            "variants": [{"signature": s, "stressed": v} for s, v in sorted(variants.items())],
        })
    return recovered


def scan(conn: psycopg.Connection, datasets: list[int],
         overridden: frozenset[str] = frozenset()) -> dict[str, Any]:
    """Find single-signature lexicon forms the trie says have two readings."""
    signatures: dict[str, set[str]] = collections.defaultdict(set)
    stressed: dict[str, str] = {}
    with conn.cursor(name="lookup_scan") as cursor:
        cursor.itersize = 200_000
        cursor.execute(
            "SELECT form_normalized, stress_signature, stressed_form FROM stress_lookup "
            "WHERE dataset_id = ANY(%s)", (datasets,))
        for form, signature, accented in cursor:
            signatures[form].add(signature)
            stressed.setdefault(form, accented)

    trie = _load_dictionary()
    recovered: list[dict[str, Any]] = []
    tally = collections.Counter()
    tally["lexicon_forms"] = len(signatures)

    for form, present in signatures.items():
        if len(present) > 1:
            tally["already_ambiguous"] += 1
            continue
        if form in overridden:
            tally["human_override"] += 1
            continue
        readings = trie_readings(trie, form)
        if readings is None:
            tally["absent_from_trie"] += 1
            continue
        if len(readings) < 2:
            tally["trie_agrees_single"] += 1
            continue

        # A tuple carrying several accents is a compound entry (Ки´їв-Льві´в),
        # not a second reading of a single word. Keeping them would fabricate
        # double-stressed variants for ordinary words.
        expected = segment_count(form)
        usable = [r for r in readings if len(r) == expected]
        if len(usable) < 2:
            tally["compound_only"] += 1
            continue

        existing = next(iter(present))
        variants: dict[str, str] = {}
        for accents in usable:
            candidate = apply_accents(form, accents)
            if validate_stress(candidate).classification == "rejected":
                continue
            try:
                signature = stress_signature(candidate)
            except NormalizationError:
                continue
            if signature != existing:
                variants.setdefault(signature, candidate)
        if not variants:
            tally["no_new_signature"] += 1
            continue

        recovered.append({
            "form": form,
            "existing_signature": existing,
            "existing_stressed": stressed[form],
            "variants": [{"signature": s, "stressed": v} for s, v in sorted(variants.items())],
        })

    recovered.sort(key=lambda r: r["form"])
    kept: list[dict[str, Any]] = []
    by_form = {r["form"]: r for r in recovered}
    surface = [(r["form"], tuple(sorted({r["existing_stressed"],
                                         *(v["stressed"] for v in r["variants"])})))
               for r in recovered]
    for row in triage.triage_forms(surface):
        tally[f"triage_{row.triage_class}"] += 1
        if row.triage_class not in KEEP_CLASSES:
            continue
        # NFD throughout: `TriagedForm.form` is NFC and would miss any key
        # containing ї or й.
        record = by_form[row.form_normalized]
        record["triage_class"] = row.triage_class
        record["needs_model"] = triage.needs_model(row)
        tally["recovered"] += 1
        tally["recovered_variants"] += len(record["variants"])
        tally["needs_model"] += int(record["needs_model"])
        kept.append(record)

    return {"tally": dict(tally), "recovered": kept}


def write_dataset(dsn: str, dataset_key: str, recovered: list[dict[str, Any]]) -> int:
    """Add recovered readings as their own unpublished dataset.

    Confidence 0.60 sits below both the Wiktionary parser's 0.95 and the LLM
    gap-fill's 0.70: a trie reading is evidence that an ambiguity *exists*, not
    an authority on which reading a given sentence wants.
    """
    written = 0
    with psycopg.connect(dsn) as conn, conn.transaction():
        row = conn.execute("SELECT id FROM import_run WHERE dataset_key=%s",
                           (dataset_key,)).fetchone()
        if row:
            dataset_id = int(row[0])
            # Delete children first, each by `dataset_id`. Deleting the lexemes
            # and letting the foreign keys cascade looks equivalent and is not:
            # every child table indexes `(dataset_id, ...)`, so a cascade
            # lookup by `lexeme_id` alone cannot use any of them and falls back
            # to a sequential scan of a 2.9M-row table *per lexeme*. Rewriting
            # six thousand rows that way ran for over half an hour; this is
            # four index scans.
            for table in ("source_ref", "stress_variant", "word_form", "lexeme"):
                conn.execute(
                    sql.SQL("DELETE FROM {} WHERE dataset_id = %s").format(
                        sql.Identifier(table)), (dataset_id,))
        else:
            dataset_id = int(conn.execute(
                """INSERT INTO import_run (dataset_key, status, dump_url, dump_sha256,
                       parser_version, normalization_version, schema_version, statistics)
                   VALUES (%s,'building','trie://ukrainian-word-stress',%s,
                           'trie-recover-1','1','007','{}')
                   RETURNING id""", (dataset_key, "0" * 64)).fetchone()[0])
        for record in recovered:
            form = record["form"]
            existing = record.get("existing_signature")
            for variant in record["variants"]:
                if existing is not None and variant["signature"] == existing:
                    continue
                stressed, signature = variant["stressed"], variant["signature"]
                key = f"{form}:{signature}"
                lexeme_id = int(conn.execute(
                    """INSERT INTO lexeme (dataset_id, lemma, lemma_normalized, stressed_lemma,
                           sense_key, source_title, source_section, confidence, natural_key)
                       VALUES (%s,%s,%s,%s,%s,%s,'TrieRecover',0.60,%s) RETURNING id""",
                    (dataset_id, form, form, stressed, signature, form,
                     f"trie:{dataset_key}:{key}")).fetchone()[0])
                word_form_id = int(conn.execute(
                    """INSERT INTO word_form (dataset_id, lexeme_id, form, form_normalized,
                           morphology_key, is_lemma, is_variant, confidence, source_rank,
                           natural_key)
                       VALUES (%s,%s,%s,%s,'',true,true,0.60,40,%s) RETURNING id""",
                    (dataset_id, lexeme_id, stressed, form,
                     f"trie:wf:{key}")).fetchone()[0])
                conn.execute(
                    """INSERT INTO stress_variant (dataset_id, word_form_id, stressed_form,
                           stress_signature, variant_type, confidence, natural_key)
                       VALUES (%s,%s,%s,%s,'primary',0.60,%s)""",
                    (dataset_id, word_form_id, stressed, signature, f"trie:sv:{key}"))
                conn.execute(
                    """INSERT INTO source_ref (dataset_id, lexeme_id, word_form_id, page_title,
                           source_kind, source_fragment, natural_key)
                       VALUES (%s,%s,%s,%s,'trie_recover',%s,%s)""",
                    (dataset_id, lexeme_id, word_form_id, form, stressed,
                     f"trie:sr:{key}"))
                written += 1
        # `stress_lookup` is a projection, not a view: without this rebuild the
        # new readings exist in the normalized tables and are invisible to
        # every reader, which looks exactly like the bug being fixed.
        build_lookup_projection(conn, dataset_id)
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database-url",
                        default="postgresql://ukstress_owner:ukstress_owner@localhost:5432/ukstress")
    parser.add_argument("--dataset", type=int, action="append", default=None,
                        help="lexicon datasets to scan; default is active + 3")
    parser.add_argument("--dataset-key", default="trie-recovered-v1")
    parser.add_argument("--overrides", type=Path,
                        default=Path("output/ml/stress_overrides.json"))
    parser.add_argument("--include-absent", type=Path,
                        help="TSV or JSON of forms missing from the lexicon "
                             "(e.g. the corpus scan's not_in_lexicon.tsv); those "
                             "the trie records ambiguously are added too")
    parser.add_argument("--out", type=Path, default=Path("output/ml/recovered_readings.json"))
    parser.add_argument("--write-db", action="store_true")
    args = parser.parse_args()

    with psycopg.connect(args.database_url) as conn:
        datasets = args.dataset
        if datasets is None:
            active = int(conn.execute(
                "SELECT dataset_id FROM active_dataset WHERE singleton").fetchone()[0])
            datasets = [active, 3]
        overridden = load_overridden(args.overrides)
        print(f"scanning datasets {datasets}; {len(overridden)} form(s) under human override",
              flush=True)
        report = scan(conn, datasets, frozenset(overridden))
        if args.include_absent:
            candidates = read_form_list(args.include_absent)
            absent = scan_absent(conn, datasets, candidates, frozenset(overridden))
            report["tally"]["absent_recovered"] = len(absent)
            report["tally"]["absent_variants"] = sum(len(r["variants"]) for r in absent)
            report["recovered"].extend(absent)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    tally = report["tally"]
    width = max(len(k) for k in tally)
    for key, value in sorted(tally.items(), key=lambda kv: -kv[1]):
        print(f"{key:<{width}}  {value:>9,}")
    print(f"\n-> {args.out}")

    if args.write_db:
        written = write_dataset(args.database_url, args.dataset_key, report["recovered"])
        print(f"wrote {written:,} variant rows to dataset '{args.dataset_key}' (unpublished)")
    else:
        print("--write-db not set; nothing written to PostgreSQL")


if __name__ == "__main__":
    main()
