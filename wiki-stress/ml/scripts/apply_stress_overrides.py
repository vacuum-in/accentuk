"""Apply human stress decisions on top of the published lexicon.

Some forms are recorded as ambiguous but should not be: `речення` carries both
`ре́чення` (a sentence) and `рече́ння` (an archaic word for an utterance), and
the second is rare enough that offering it costs more than it gains — the model
picked it with margin 10.2 for "коротке речення" because the gloss "a short
expression" matched the adjective, not the meaning.

Overrides live in a file, not in a one-off SQL statement, for two reasons: a
manual edit to a published dataset disappears the next time the dataset is
rebuilt from staging, and a reviewer needs to see which stresses were decided
by a person rather than derived from a source.

Each override names the form, the stress to keep, and why. Applying it deletes
the other variants for that form and rebuilds only the affected rows of
`stress_lookup`, so the form stops being reported as ambiguous and stops
reaching the model at all.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import psycopg
from ukstress.normalizer import canonical_stressed_form, lookup_key, stress_signature


def load_overrides(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def apply(dsn: str, overrides: list[dict[str, Any]], dataset_id: int | None = None) -> dict:
    applied: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    with psycopg.connect(dsn) as conn, conn.transaction():
        if dataset_id is None:
            row = conn.execute("SELECT dataset_id FROM active_dataset WHERE singleton").fetchone()
            if row is None:
                raise RuntimeError("no active dataset")
            dataset_id = int(row[0])

        for override in overrides:
            form = lookup_key(override["form"])
            keep = canonical_stressed_form(override["keep"])
            if lookup_key(keep) != form:
                skipped.append({**override, "reason": "keep does not spell the form"})
                continue

            present = conn.execute(
                "SELECT DISTINCT stressed_form FROM stress_lookup "
                "WHERE dataset_id=%s AND form_normalized=%s", (dataset_id, form)).fetchall()
            spellings = {str(r[0]) for r in present}
            if keep not in spellings:
                # The kept spelling is not in the lexicon at all, so this is a
                # correction rather than a choice between recorded readings.
                # That is a stronger claim than picking one of two sourced
                # variants, so it must be asked for explicitly.
                if not override.get("replace"):
                    skipped.append({**override, "reason":
                                    f"kept spelling absent; have {sorted(spellings)} "
                                    "(set \"replace\": true to correct it)"})
                    continue
                signature = stress_signature(keep)
                conn.execute(
                    """
                    UPDATE stress_variant v SET stressed_form=%s, stress_signature=%s
                    FROM word_form w
                    WHERE v.word_form_id=w.id AND v.dataset_id=%s AND w.form_normalized=%s
                    """, (keep, signature, dataset_id, form))
                conn.execute(
                    "UPDATE stress_lookup SET stressed_form=%s, stress_signature=%s "
                    "WHERE dataset_id=%s AND form_normalized=%s",
                    (keep, signature, dataset_id, form))
                applied.append({**override, "variants_removed": 0,
                                "was": sorted(spellings), "now": keep, "mode": "replaced"})
                continue
            if len(spellings) < 2:
                skipped.append({**override, "reason": "already unambiguous"})
                continue

            # Remove the variants at the source, then the projection rows, so a
            # later rebuild from `stress_variant` reproduces the same result.
            removed = conn.execute(
                """
                DELETE FROM stress_variant v
                USING word_form w
                WHERE v.word_form_id = w.id AND v.dataset_id = %s
                  AND w.form_normalized = %s AND v.stressed_form <> %s
                """, (dataset_id, form, keep)).rowcount
            conn.execute(
                "DELETE FROM stress_lookup WHERE dataset_id=%s AND form_normalized=%s "
                "AND stressed_form <> %s", (dataset_id, form, keep))
            applied.append({**override, "variants_removed": removed,
                            "was": sorted(spellings), "now": keep})

    return {"applied": applied, "skipped": skipped, "dataset_id": dataset_id}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overrides", type=Path, default=Path("output/ml/stress_overrides.json"))
    parser.add_argument("--database-url",
                        default="postgresql://ukstress_owner:ukstress_owner@localhost:5432/ukstress")
    parser.add_argument("--dataset-id", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    overrides = load_overrides(args.overrides)
    if not overrides:
        print(f"no overrides in {args.overrides}")
        return
    if args.dry_run:
        for override in overrides:
            print(f"  would keep {override['keep']!r} for {override['form']!r}: {override.get('reason','')}")
        return

    result = apply(args.database_url, overrides, args.dataset_id)
    print(f"dataset {result['dataset_id']}")
    for entry in result["applied"]:
        print(f"  {entry['form']:<16} {entry['was']} -> {entry['now']}  "
              f"({entry.get('mode','kept')}"
              + (f", {entry['variants_removed']} variants removed" if entry['variants_removed'] else "")
              + ")")
    for entry in result["skipped"]:
        print(f"  SKIPPED {entry['form']:<12} {entry['reason']}")


if __name__ == "__main__":
    main()
