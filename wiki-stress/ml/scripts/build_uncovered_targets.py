"""List the ambiguous forms that need glosses before they can reach the model.

A form belongs here when the lexicon reports more than one stress signature for
it, the surface triage says only sentence context can decide between them, and
the serving manifest does not list it. That is precisely the class the gold
evaluation attributed to `outside_coverage` — rows where the pipeline holds
both readings and serves neither.

Each variant carries the source trie's own tag sets for that accent position.
A pilot without them had the labeller derive `коли́` from `кола` ("circle"),
which is the reading the trie assigns to `ко́ли`; the tags say `коли́` is an
adverb, a conjunction, an imperative, and the plural of `кіл`, and none of them
is a circle. Handing the labeller the morphology it would otherwise have to
guess is the difference between describing a reading and inventing one.

Ordering is by priority band, then by corpus frequency inside the band.

Band 1 is the forms `run_recover_readings.py` just made ambiguous. They are the
one group the pipeline is *worse* at than before the recovery: they used to be
answered by the dictionary at 0.50 on the gold set and now route to a model
that will not take them, so they must be covered or the recovery is a
regression. Band 2 is everything else, most frequent first.

Frequency ordering within a band is not cosmetic, and it is what makes the list
finishable.
1,030 of these forms appear at all in the 3M-token corpus scan, and the most
frequent 200 of them carry 21.1% of that scan's uncovered tokens against 22.5%
for all 1,030 — so the running-text value is essentially exhausted in the first
few hundred calls. The remaining ten thousand are rare forms that matter to a
homograph benchmark and almost never to running text, which is the right order
to spend an annotation budget in and the right thing to say about a run that is
stopped early.
"""

from __future__ import annotations

import argparse
import json
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import psycopg
from ukrainian_word_stress.stressify_ import (
    _load_dictionary,
    _parse_dictionary_value,
    _trie_value,
)
from ukstress.normalizer import ACUTE, NormalizationError, canonical_stressed_form, stress_signature

from ukstress_ml import triage

#: Order tags are shown in, so a labeller reading many of them sees the same
#: shape every time rather than the trie's insertion order.
TAG_ORDER = ("upos", "VerbForm", "Number", "Case", "Gender", "Animacy")


def readings_by_signature(trie: Any, form: str) -> dict[str, list[str]]:
    """Group the trie's tag sets by the stress signature their accents produce."""
    # `form_normalized` comes out of the database in NFD; the trie's keys are
    # NFC. Passing the decomposed string silently missed every form containing
    # `ї`, `й` or any other composable letter -- the same defect that kept
    # `украї́ни` and `діте́й` out of the corrections dataset. The accent
    # positions the trie returns index the composed string, so callers must
    # place accents into it, not into `form`.
    form = unicodedata.normalize("NFC", form)
    values = _trie_value(trie, form)
    if values is None:
        return {}
    grouped: dict[str, list[str]] = {}
    for tags, accents in _parse_dictionary_value(values[0]):
        stressed = form
        for position in sorted(accents, reverse=True):
            stressed = stressed[:position] + ACUTE + stressed[position:]
        try:
            signature = stress_signature(canonical_stressed_form(stressed))
        except NormalizationError:
            continue
        features = dict(tag.split("=", 1) for tag in tags if "=" in tag)
        rendered = " ".join(features[name] for name in TAG_ORDER if name in features)
        bucket = grouped.setdefault(signature, [])
        if rendered and rendered not in bucket:
            bucket.append(rendered)
    return grouped


def load_frequencies(paths: list[Path]) -> Counter[str]:
    """Read corpus counts from the scan's TSVs or from a JSON form/tokens list.

    Keys are normalised to NFD, because the corpus scan writes what it read and
    every key these targets are matched against is `lookup_key` output.
    """
    counts: Counter[str] = Counter()
    for path in paths:
        if not path.exists():
            continue
        if path.suffix == ".tsv":
            for line in path.read_text(encoding="utf-8").splitlines():
                count, _, form = line.partition("\t")
                if count.isdigit() and form:
                    counts[unicodedata.normalize("NFD", form)] += int(count)
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload if isinstance(payload, list) else payload.get("forms", [])
        for row in rows:
            if isinstance(row, dict) and "form" in row:
                counts[unicodedata.normalize("NFD", str(row["form"]))] += int(
                    row.get("tokens", row.get("count", 0)))
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database-url",
                        default="postgresql://ukstress_owner:ukstress_owner@localhost:5432/ukstress")
    parser.add_argument("--dataset", type=int, action="append", default=None)
    parser.add_argument("--manifest", type=Path,
                        default=Path("output/ml/serving_manifest_expanded.json"))
    parser.add_argument("--frequencies", type=Path, action="append",
                        default=[Path("output/ml/corpus_3m/outside_coverage.tsv"),
                                 Path("output/ml/corpus_3m/model_abstained.tsv"),
                                 Path("output/ml/high_freq_uncovered.json")])
    parser.add_argument("--recovered", type=Path,
                        default=Path("output/ml/recovered_readings.json"),
                        help="forms this cycle made ambiguous; they sort first")
    parser.add_argument("--out", type=Path, default=Path("output/ml/uncovered_targets.json"))
    args = parser.parse_args()

    with psycopg.connect(args.database_url) as conn:
        datasets = args.dataset
        if datasets is None:
            active = int(conn.execute(
                "SELECT dataset_id FROM active_dataset WHERE singleton").fetchone()[0])
            datasets = [active, 3, 4]
        rows = conn.execute(
            """
            SELECT form_normalized,
                   array_agg(DISTINCT stressed_form ORDER BY stressed_form),
                   array_agg(DISTINCT stress_signature ORDER BY stress_signature)
            FROM stress_lookup
            WHERE dataset_id = ANY(%s)
            GROUP BY form_normalized
            HAVING count(DISTINCT stress_signature) > 1
            ORDER BY form_normalized
            """, (datasets,)).fetchall()

    covered = set(json.loads(args.manifest.read_text(encoding="utf-8"))["forms"])
    frequencies = load_frequencies(args.frequencies)
    recovered: set[str] = set()
    if args.recovered.exists():
        recovered = {str(r["form"]) for r
                     in json.loads(args.recovered.read_text(encoding="utf-8"))["recovered"]}
    trie = _load_dictionary()

    surface = [(str(form), tuple(str(v) for v in stressed)) for form, stressed, _ in rows]
    signatures = {str(form): [str(s) for s in sigs] for form, _, sigs in rows}
    stressed_by_form = {str(form): [str(v) for v in stressed] for form, stressed, _ in rows}

    targets: list[dict[str, Any]] = []
    tally: Counter[str] = Counter()
    for row in triage.triage_forms(surface):
        # `TriagedForm.form` is NFC for reading; every key downstream — the
        # lookup table, the manifest, `lookup_key` — is NFD, and the two differ
        # for any form containing ї or й.
        form = row.form_normalized
        tally[row.triage_class] += 1
        if not triage.needs_model(row):
            continue
        tally["needs_model"] += 1
        if form in covered:
            tally["already_in_manifest"] += 1
            continue
        variants = stressed_by_form[form]
        if len(variants) != len(signatures[form]):
            # One signature spelled two ways (case, apostrophe) would give the
            # model two candidates that are the same answer.
            tally["signature_variant_mismatch"] += 1
            continue
        if any("|" in sig for sig in signatures[form]):
            # A signature naming two accents in one token is the wordlist's
            # "either stress is acceptable" notation, not a sense to gloss.
            tally["free_variation_notation"] += 1
            continue
        grouped = readings_by_signature(trie, form)
        targets.append({
            "form": form,
            "triage_class": row.triage_class,
            "band": 1 if form in recovered else 2,
            "tokens": frequencies.get(form, 0),
            "variants": [{"id": chr(ord("a") + i), "stressed": s, "signature": sig,
                          "readings": grouped.get(sig, [])}
                         for i, (s, sig) in enumerate(zip(variants, signatures[form],
                                                          strict=True))],
        })

    targets.sort(key=lambda t: (t["band"], -t["tokens"], t["form"]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(targets, ensure_ascii=False, indent=1), encoding="utf-8")

    width = max(len(k) for k in tally)
    for key, value in sorted(tally.items(), key=lambda kv: -kv[1]):
        print(f"{key:<{width}}  {value:>8,}")
    print(f"\ntargets needing glosses: {len(targets):,}  -> {args.out}")
    print(f"  band 1, made ambiguous this cycle: {sum(1 for t in targets if t['band'] == 1):,}")
    print(f"  seen in the corpus scan          : {sum(1 for t in targets if t['tokens']):,}")


if __name__ == "__main__":
    main()
