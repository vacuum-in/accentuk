"""Import the immutable contextual-model candidate inventory into PostgreSQL."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import psycopg

from ukstress.normalizer import lookup_key, stress_signature


def import_serving_inventory(database_url: str, manifest_path: Path) -> dict[str, object]:
    manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    inventory_hash = str(manifest["inventory_hash"])
    model_version = str(manifest["model_version"])
    forms = manifest.get("forms")
    if not isinstance(forms, dict) or not forms:
        raise ValueError("serving manifest has no forms")

    rows: list[tuple[str, str, str, str, str, str, str]] = []
    represented_forms: set[str] = set()
    represented_signatures: set[tuple[str, str]] = set()
    for raw_form, raw_entry in forms.items():
        form = lookup_key(str(raw_form))
        entry = dict(raw_entry)
        group_id = str(entry["group_id"])
        candidates = entry.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise ValueError(f"serving manifest form {form!r} has no candidates")
        for raw_candidate in candidates:
            candidate = dict(raw_candidate)
            stressed = str(candidate["stressed"])
            signature = str(candidate["signature"])
            if lookup_key(stressed) != form:
                raise ValueError(f"stressed spelling does not match form {form!r}")
            if stress_signature(stressed) != signature:
                raise ValueError(f"stress signature does not match form {form!r}")
            rows.append(
                (
                    inventory_hash,
                    model_version,
                    form,
                    signature,
                    stressed,
                    group_id,
                    str(candidate["sense_id"]),
                )
            )
            represented_forms.add(form)
            represented_signatures.add((form, signature))

    with psycopg.connect(database_url) as connection:
        connection.execute(
            "DELETE FROM contextual_stress_candidate WHERE inventory_hash = %s",
            (inventory_hash,),
        )
        with connection.cursor() as cursor:
            cursor.executemany(
                """
                INSERT INTO contextual_stress_candidate (
                    inventory_hash, model_version, form_normalized, stress_signature,
                    stressed_form, group_id, sense_id
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                rows,
            )

    return {
        "inventory_hash": inventory_hash,
        "model_version": model_version,
        "forms": len(represented_forms),
        "signatures": len(represented_signatures),
        "candidate_senses": len(rows),
    }
