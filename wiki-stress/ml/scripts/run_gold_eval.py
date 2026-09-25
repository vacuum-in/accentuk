"""Evaluate the full pipeline against an external gold set.

Every accuracy figure before this one was measured against *silver* labels —
two models agreeing — and a symmetric control put their residual error near 2%,
which caps what any silver-scored number can prove. This scores the pipeline
against labels it had no part in producing.

The pipeline is run exactly as it serves: tier 1 the PostgreSQL lookup, tier 2
the morphological parse, tier 3 the cross-encoder. Each row is attributed to
the tier that answered it, so a wrong answer can be traced to the component
responsible rather than to "the model".

**This script collects evidence; it does not apply policy.** The abstention
threshold and the choice of what to serve when nothing decides are serving
*settings*, and baking them in here meant every question about them cost
another GPU pass over the whole set — which is how a fallback that scores below
chance survived unmeasured. Every row is written to `decisions.json` with its
candidates, their lexicon confidences, the morphological parse, and the model's
full score vector. `score_gold.py` turns that into an accuracy under a given
policy, so a threshold sweep is a re-read of a file.

Comparison ignores letter case and normalises Unicode, because a gold file may
capitalise a sentence-initial target that the lexicon stores lowercase; it does
not ignore which vowel carries the acute, which is the whole question.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import re
import subprocess
import sys
import unicodedata
from pathlib import Path
from typing import Any

import psycopg
from ukstress.normalizer import lookup_key

from ukstress_ml.morphology import MorphologyTier, apply_accents
from ukstress_ml.morphology import resolve as morph_resolve
from ukstress_ml.serving import ContextualStressModel, Target

VOWELS = frozenset("аеєиіїоуюя")


def same_stress(first: str, second: str) -> bool:
    return unicodedata.normalize("NFD", first).lower() == \
        unicodedata.normalize("NFD", second).lower()


def find_span(sentence: str, target: str) -> tuple[int, int] | None:
    key = lookup_key(target)
    for match in re.finditer(r"[^\W\d_]+(?:['’ʼ-][^\W\d_]+)*", sentence, re.UNICODE):
        if lookup_key(match.group()) == key:
            return match.start(), match.end()
    return None


def read_variants(conn: psycopg.Connection, datasets: list[int],
                  keys: list[str]) -> dict[str, list[dict[str, Any]]]:
    """Return each form's candidate readings, best-first.

    Ordering is the lexicon's own: `stress_lookup` is read confidence-first and
    then by `source_rank`, so "the dictionary's answer" here means the same row
    the API's exact-lookup path would serve, not an arbitrary pick. The
    recovered trie readings sit at confidence 0.60 by construction, which is
    what keeps them from displacing a curated stress as the default.
    """
    rows = conn.execute(
        "SELECT form_normalized, stress_signature, min(stressed_form), "
        "       max(confidence), min(source_rank) "
        "FROM stress_lookup "
        "WHERE dataset_id = ANY(%s) AND form_normalized = ANY(%s) "
        "GROUP BY form_normalized, stress_signature", (datasets, keys)).fetchall()
    variants: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for form, signature, stressed, confidence, rank in rows:
        variants[str(form)].append({"signature": str(signature), "stressed": str(stressed),
                                    "confidence": float(confidence), "source_rank": int(rank)})
    for candidates in variants.values():
        candidates.sort(key=lambda c: (-c["confidence"], c["source_rank"], c["signature"]))
    return variants


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sentences", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--database-url",
                        default="postgresql://ukstress_owner:ukstress_owner@localhost:5432/ukstress")
    parser.add_argument("--model", type=Path, default=Path("output/ml/models/v8-generated2"))
    parser.add_argument("--manifest", type=Path,
                        default=Path("output/ml/serving_manifest_expanded.json"))
    parser.add_argument("--extra-dataset", type=int, action="append", default=[3])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, default=Path("output/ml/gold_eval"))
    args = parser.parse_args()

    rows = {r["id"]: r for r in csv.DictReader(args.sentences.open(encoding="utf-8-sig"))}
    gold = {r["id"]: r for r in csv.DictReader(args.gold.open(encoding="utf-8-sig"))}
    ids = [i for i in rows if i in gold]
    print(f"rows: {len(rows):,}  gold: {len(gold):,}  evaluable: {len(ids):,}", flush=True)

    model = ContextualStressModel(args.model, args.manifest, backend="torch", device=args.device)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))["forms"]
    morphology = MorphologyTier()
    reading_cache: dict[str, list] = {}

    with psycopg.connect(args.database_url) as conn:
        active = int(conn.execute(
            "SELECT dataset_id FROM active_dataset WHERE singleton").fetchone()[0])
        datasets = [active, *args.extra_dataset]
        keys = sorted({lookup_key(rows[i]["target"]) for i in ids})
        found = read_variants(conn, datasets, keys)

    decisions: list[dict[str, Any]] = []
    pending: list[tuple[int, Target]] = []

    for row_id in ids:
        row = rows[row_id]
        sentence, target = row["sentence"], row["target"]
        key = lookup_key(target)
        span = find_span(sentence, target)
        candidates = found.get(key, [])
        entry = manifest.get(key)
        record: dict[str, Any] = {
            "id": row_id,
            "target": target,
            "form": key,
            "gold": gold[row_id]["target_stressed"],
            "sense": gold[row_id].get("sense", ""),
            "sentence": sentence,
            "candidates": candidates,
            "manifest_default": (entry or {}).get("default_signature"),
            "monosyllabic": sum(1 for c in target.lower() if c in VOWELS) <= 1,
            "span_found": span is not None,
            "morphology": None,
            "model": None,
        }

        if span is None or record["monosyllabic"] or len(candidates) < 2:
            decisions.append(record)
            continue

        # Morphology is recorded for every ambiguous row, not only the rows the
        # model cannot take. It is a *fallback* worth comparing against the
        # dictionary default, and that comparison is impossible if it is only
        # ever run where the model already failed.
        if key not in reading_cache:
            reading_cache[key] = morphology.readings(key)
        if reading_cache[key]:
            try:
                parse = morphology.parse(sentence).get(span)
            except Exception:  # noqa: BLE001
                parse = None
            if parse:
                resolution = morph_resolve(reading_cache[key], parse[0], parse[1])
                if resolution is not None:
                    record["morphology"] = apply_accents(target, resolution.accents)

        signatures = [c["signature"] for c in candidates]
        if entry is not None and sorted(entry["signatures"]) == sorted(signatures):
            pending.append((len(decisions), Target(sentence, span[0], span[1], key,
                                                   tuple(sorted(signatures)))))
        decisions.append(record)

    if pending:
        print(f"model targets: {len(pending):,}", flush=True)
        resolved = model.resolve([t for _, t in pending])
        for (index, _), decision in zip(pending, resolved, strict=True):
            decisions[index]["model"] = {
                "signature": decision["signature"],
                "margin": round(float(decision["margin"]), 4),
                "scores": {k: round(float(v), 4) for k, v in (decision.get("scores") or {}).items()},
                "status": decision["status"],
            }

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "decisions.json"
    path.write_text(json.dumps(decisions, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n-> {path}")

    scorer = Path(__file__).with_name("score_gold.py")
    subprocess.run([sys.executable, str(scorer), "--decisions", str(path),
                    "--out", str(args.out)], check=True)


if __name__ == "__main__":
    main()
