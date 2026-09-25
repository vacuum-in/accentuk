"""Read-only PostgreSQL statistics, diagnostics, and data exports."""

from __future__ import annotations

import csv
import json
import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, cast

import psycopg


def database_statistics(database_url: str, dataset_id: int | None = None) -> dict[str, Any]:
    with psycopg.connect(database_url) as connection:
        resolved = dataset_id
        if resolved is None:
            active = connection.execute(
                "SELECT dataset_id FROM active_dataset WHERE singleton"
            ).fetchone()
            if active is None:
                raise ValueError("no active dataset")
            resolved = cast(tuple[int], active)[0]
        counts = {}
        for table in (
            "lexeme",
            "word_form",
            "stress_variant",
            "source_ref",
            "stress_lookup",
            "parse_error",
            "rejected_form",
            "unhandled_template",
        ):
            row = connection.execute(
                f"SELECT count(*) FROM {table} WHERE dataset_id = %s",  # noqa: S608
                (resolved,),
            ).fetchone()
            counts[table] = 0 if row is None else int(row[0])
        status = connection.execute(
            "SELECT dataset_key, status, statistics FROM import_run WHERE id = %s",
            (resolved,),
        ).fetchone()
        if status is None:
            raise ValueError("dataset does not exist")
    return {
        "dataset_id": resolved,
        "dataset_key": status[0],
        "status": status[1],
        "counts": counts,
        "import_statistics": status[2],
    }


def unhandled_template_report(
    database_url: str, dataset_id: int | None = None
) -> list[dict[str, Any]]:
    with psycopg.connect(database_url) as connection:
        rows = connection.execute(
            """
            SELECT normalized_name, occurrences, sample_page_title, sample_invocation
            FROM unhandled_template
            WHERE dataset_id = COALESCE(
                %s, (SELECT dataset_id FROM active_dataset WHERE singleton)
            )
            ORDER BY occurrences DESC, normalized_name
            """,
            (dataset_id,),
        ).fetchall()
    return [
        {
            "normalized_name": row[0],
            "occurrences": row[1],
            "sample_page_title": row[2],
            "sample_invocation": row[3],
        }
        for row in rows
    ]


def export_rows(
    database_url: str,
    *,
    dataset_id: int | None = None,
    best_only: bool = False,
) -> list[dict[str, Any]]:
    """Read deterministic rows; best-only ranks a projection without mutating it."""
    rank_filter = "WHERE candidate_rank = 1" if best_only else ""
    query = f"""
        WITH ranked AS (
            SELECT form_normalized, stressed_form, stress_signature,
                   lemma_normalized, stressed_lemma, part_of_speech,
                   grammatical_tags, is_lemma, is_variant, is_obsolete,
                   confidence, source_rank,
                   row_number() OVER (
                       PARTITION BY form_normalized
                       ORDER BY is_obsolete, confidence DESC, source_rank,
                                is_lemma DESC, stressed_form
                   ) AS candidate_rank
            FROM stress_lookup
            WHERE dataset_id = COALESCE(
                %s, (SELECT dataset_id FROM active_dataset WHERE singleton)
            )
        )
        SELECT form_normalized, stressed_form, stress_signature,
               lemma_normalized, stressed_lemma, part_of_speech,
               grammatical_tags, is_lemma, is_variant, is_obsolete,
               confidence, source_rank
        FROM ranked
        {rank_filter}
        ORDER BY form_normalized, is_obsolete, confidence DESC,
                 source_rank, is_lemma DESC, stressed_form
    """
    with psycopg.connect(database_url) as connection:
        rows = connection.execute(query, (dataset_id,)).fetchall()
    fields = (
        "form_normalized",
        "stressed_form",
        "stress_signature",
        "lemma_normalized",
        "stressed_lemma",
        "part_of_speech",
        "grammatical_tags",
        "is_lemma",
        "is_variant",
        "is_obsolete",
        "confidence",
        "source_rank",
    )
    return [dict(zip(fields, row, strict=True)) for row in rows]


def _atomic_replace(path: Path) -> None:
    os.replace(path.with_suffix(path.suffix + ".part"), path)


def write_export(
    path: Path, rows: Iterable[Mapping[str, Any]], export_format: str
) -> int:
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    if export_format == "jsonl":
        with temporary.open("w", encoding="utf-8") as output:
            for row in materialized:
                output.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
    elif export_format == "tsv":
        fields = list(materialized[0]) if materialized else ["form_normalized", "stressed_form"]
        with temporary.open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=fields, delimiter="\t")
            writer.writeheader()
            for row in materialized:
                writer.writerow(
                    {
                        key: json.dumps(value, ensure_ascii=False)
                        if isinstance(value, (list, dict))
                        else value
                        for key, value in row.items()
                    }
                )
    elif export_format == "tts":
        with temporary.open("w", encoding="utf-8") as output:
            for row in materialized:
                output.write(f"{row['form_normalized']}\t{row['stressed_form']}\n")
    else:
        raise ValueError(f"unsupported export format: {export_format}")
    _atomic_replace(path)
    return len(materialized)
