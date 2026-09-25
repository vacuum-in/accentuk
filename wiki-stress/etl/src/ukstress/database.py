"""Python-owned PostgreSQL migration and import primitives."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import psycopg
from psycopg import sql
from psycopg.types.json import Jsonb

from ukstress.staging import iter_jsonl_zst, read_jsonl_zst

logger = logging.getLogger("ukstress.database")

_STAGING_TABLES = frozenset(
    {
        "staging_lexeme",
        "staging_word_form",
        "staging_stress_variant",
        "staging_source_ref",
    }
)


def applied_migrations(
    connection: psycopg.Connection[tuple[object, ...]],
) -> set[str]:
    exists = connection.execute(
        "SELECT to_regclass('public.schema_migration') IS NOT NULL"
    ).fetchone()
    if exists is None or not bool(exists[0]):
        return set()
    return {
        str(row[0])
        for row in connection.execute("SELECT version FROM schema_migration").fetchall()
    }


def apply_migrations(
    database_url: str, migrations_dir: Path, *, check_only: bool = False
) -> list[str]:
    """Apply append-only SQL files in lexical order, one transaction per migration."""
    paths = sorted(migrations_dir.glob("*.sql"))
    if not paths:
        raise ValueError(f"no SQL migrations found in {migrations_dir}")
    applied: list[str] = []
    with psycopg.connect(database_url) as connection:
        known = applied_migrations(connection)
        for path in paths:
            version = path.stem
            if version in known:
                continue
            if check_only:
                applied.append(version)
                continue
            with connection.transaction():
                connection.execute(path.read_text(encoding="utf-8"))
                recorded = connection.execute(
                    "SELECT 1 FROM schema_migration WHERE version = %s", (version,)
                ).fetchone()
                if recorded is None:
                    raise RuntimeError(f"migration {version} did not record itself")
            applied.append(version)
    if not check_only and applied:
        logger.info("migrations applied", extra={"versions": applied})
    return applied


def copy_staging_records(
    database_url: str,
    table: str,
    dataset_id: int,
    records: Iterable[Mapping[str, Any]],
) -> int:
    """Stream records through PostgreSQL COPY without buffering or row inserts."""
    if table not in _STAGING_TABLES:
        raise ValueError(f"unsupported staging table: {table}")
    statement = sql.SQL("COPY {} (dataset_id, natural_key, payload) FROM STDIN").format(
        sql.Identifier(table)
    )
    copied = 0
    with psycopg.connect(database_url) as connection:
        with connection.cursor().copy(statement) as copy:
            for record in records:
                natural_key = str(record["natural_key"])
                copy.write_row((dataset_id, natural_key, Jsonb(dict(record))))
                copied += 1
    return copied


def validate_staging(
    connection: psycopg.Connection[tuple[object, ...]], dataset_id: int
) -> dict[str, int]:
    """Validate natural-key consistency before normalized records are merged."""
    counts: dict[str, int] = {}
    for table in sorted(_STAGING_TABLES):
        query = sql.SQL(
            """
            SELECT count(*),
                   count(*) FILTER (
                       WHERE payload->>'natural_key' IS DISTINCT FROM natural_key
                   )
            FROM {}
            WHERE dataset_id = %s
            """
        ).format(sql.Identifier(table))
        row = connection.execute(query, (dataset_id,)).fetchone()
        if row is None:
            raise RuntimeError(f"failed to validate {table}")
        count, invalid = cast(tuple[int, int], row)
        if invalid:
            raise ValueError(f"{table} contains {invalid} inconsistent natural keys")
        counts[table] = count
    return counts


def merge_staging(connection: psycopg.Connection[tuple[object, ...]], dataset_id: int) -> None:
    """Merge validated staging JSON into normalized dataset-scoped tables."""
    connection.execute(
        """
        INSERT INTO lexeme (
            dataset_id, natural_key, lemma, lemma_normalized, stressed_lemma,
            part_of_speech_id, sense_key, source_title, source_section,
            is_multiword, is_obsolete, confidence
        )
        SELECT s.dataset_id, s.natural_key, s.payload->>'lemma',
               s.payload->>'lemma_normalized', s.payload->>'stressed_lemma',
               p.id, COALESCE(s.payload->>'sense_key', ''),
               s.payload->>'source_title', COALESCE(s.payload->>'source_section', ''),
               COALESCE((s.payload->>'is_multiword')::boolean, false),
               COALESCE((s.payload->>'is_obsolete')::boolean, false),
               (s.payload->>'confidence')::real
        FROM staging_lexeme s
        LEFT JOIN part_of_speech p ON p.code = s.payload->>'part_of_speech'
        WHERE s.dataset_id = %s
        ON CONFLICT (dataset_id, natural_key) DO NOTHING
        """,
        (dataset_id,),
    )
    connection.execute(
        """
        INSERT INTO word_form (
            dataset_id, natural_key, lexeme_id, form, form_normalized,
            grammatical_tags, morphology_key, is_lemma, is_variant,
            confidence, source_rank
        )
        SELECT s.dataset_id, s.natural_key, l.id, s.payload->>'form',
               s.payload->>'form_normalized',
               ARRAY(
                   SELECT jsonb_array_elements_text(
                       COALESCE(s.payload->'grammatical_tags', '[]'::jsonb)
                   )
               ),
               COALESCE(s.payload->>'morphology_key', ''),
               COALESCE((s.payload->>'is_lemma')::boolean, false),
               COALESCE((s.payload->>'is_variant')::boolean, false),
               (s.payload->>'confidence')::real,
               (s.payload->>'source_rank')::smallint
        FROM staging_word_form s
        JOIN lexeme l
          ON l.dataset_id = s.dataset_id
         AND l.natural_key = s.payload->>'lexeme_natural_key'
        WHERE s.dataset_id = %s
        ON CONFLICT (dataset_id, natural_key) DO NOTHING
        """,
        (dataset_id,),
    )
    connection.execute(
        """
        INSERT INTO stress_variant (
            dataset_id, natural_key, word_form_id, stressed_form,
            stress_signature, variant_type, confidence
        )
        SELECT s.dataset_id, s.natural_key, w.id, s.payload->>'stressed_form',
               s.payload->>'stress_signature',
               COALESCE(s.payload->>'variant_type', 'primary'),
               (s.payload->>'confidence')::real
        FROM staging_stress_variant s
        JOIN word_form w
          ON w.dataset_id = s.dataset_id
         AND w.natural_key = s.payload->>'word_form_natural_key'
        WHERE s.dataset_id = %s
        ON CONFLICT (dataset_id, natural_key) DO NOTHING
        """,
        (dataset_id,),
    )
    connection.execute(
        """
        INSERT INTO source_ref (
            dataset_id, natural_key, lexeme_id, word_form_id, page_id,
            revision_id, page_title, source_kind, source_section,
            source_fragment, source_offset
        )
        SELECT s.dataset_id, s.natural_key, l.id, w.id,
               (s.payload->>'page_id')::bigint,
               (s.payload->>'revision_id')::bigint,
               s.payload->>'page_title', s.payload->>'source_kind',
               s.payload->>'source_section', s.payload->>'source_fragment',
               (s.payload->>'source_offset')::integer
        FROM staging_source_ref s
        LEFT JOIN lexeme l
          ON l.dataset_id = s.dataset_id
         AND l.natural_key = s.payload->>'lexeme_natural_key'
        LEFT JOIN word_form w
          ON w.dataset_id = s.dataset_id
         AND w.natural_key = s.payload->>'word_form_natural_key'
        WHERE s.dataset_id = %s
        ON CONFLICT (dataset_id, natural_key) DO NOTHING
        """,
        (dataset_id,),
    )


def build_lookup_projection(
    connection: psycopg.Connection[tuple[object, ...]], dataset_id: int
) -> int:
    """Rebuild one immutable dataset's flattened read projection."""
    connection.execute("DELETE FROM stress_lookup WHERE dataset_id = %s", (dataset_id,))
    inserted = connection.execute(
        """
        INSERT INTO stress_lookup (
            dataset_id, form_normalized, stressed_form, stress_signature,
            lemma_normalized, stressed_lemma, part_of_speech,
            grammatical_tags, lexeme_id, word_form_id, is_lemma, is_variant,
            is_obsolete, confidence, source_rank
        )
        SELECT w.dataset_id, w.form_normalized, v.stressed_form,
               v.stress_signature, l.lemma_normalized, l.stressed_lemma,
               p.code, w.grammatical_tags, l.id, w.id, w.is_lemma,
               w.is_variant OR v.variant_type <> 'primary',
               l.is_obsolete, LEAST(l.confidence, w.confidence, v.confidence),
               w.source_rank
        FROM word_form w
        JOIN lexeme l ON l.id = w.lexeme_id AND l.dataset_id = w.dataset_id
        JOIN stress_variant v
          ON v.word_form_id = w.id AND v.dataset_id = w.dataset_id
        LEFT JOIN part_of_speech p ON p.id = l.part_of_speech_id
        WHERE w.dataset_id = %s
        ON CONFLICT DO NOTHING
        """,
        (dataset_id,),
    ).rowcount
    expected_row = connection.execute(
        """
        SELECT count(*)
        FROM word_form w
        JOIN stress_variant v
          ON v.word_form_id = w.id AND v.dataset_id = w.dataset_id
        WHERE w.dataset_id = %s
        """,
        (dataset_id,),
    ).fetchone()
    expected = None if expected_row is None else cast(tuple[int], expected_row)[0]
    if inserted != expected:
        raise RuntimeError("lookup projection row count does not match normalized records")
    return inserted


def finalize_dataset(
    connection: psycopg.Connection[tuple[object, ...]], dataset_id: int
) -> None:
    """Ensure hot-path indexes exist after load and refresh planner statistics."""
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS stress_lookup_exact_idx
        ON stress_lookup (
            dataset_id, form_normalized, is_obsolete, confidence DESC, source_rank
        )
        INCLUDE (
            stressed_form, stress_signature, lemma_normalized, stressed_lemma,
            part_of_speech, grammatical_tags, lexeme_id, word_form_id,
            is_lemma, is_variant
        )
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS stress_lookup_lemma_idx
        ON stress_lookup (dataset_id, lemma_normalized)
        INCLUDE (form_normalized, stressed_form, grammatical_tags, confidence)
        """
    )
    connection.execute(sql.SQL("ANALYZE {}").format(sql.Identifier("stress_lookup")))
    connection.execute(
        "UPDATE import_run SET status = 'validated' WHERE id = %s AND status = 'building'",
        (dataset_id,),
    )


@dataclass(frozen=True)
class QualityGates:
    minimum_lookup_rows: int = 1
    minimum_relative_rows: float = 0.8
    maximum_rejected_ratio: float = 0.2


class QualityGateError(RuntimeError):
    pass


def enforce_quality_gates(
    connection: psycopg.Connection[tuple[object, ...]],
    dataset_id: int,
    gates: QualityGates,
) -> dict[str, float]:
    row = connection.execute(
        """
        SELECT
            (SELECT count(*) FROM stress_lookup WHERE dataset_id = %s),
            (SELECT count(*) FROM rejected_form WHERE dataset_id = %s),
            (
                SELECT count(*)
                FROM stress_lookup
                WHERE dataset_id = (
                    SELECT dataset_id FROM active_dataset WHERE singleton
                )
            )
        """,
        (dataset_id, dataset_id),
    ).fetchone()
    if row is None:
        raise RuntimeError("quality statistics query returned no row")
    lookup_rows, rejected_rows, active_rows = cast(tuple[int, int, int], row)
    total_candidates = lookup_rows + rejected_rows
    rejected_ratio = rejected_rows / total_candidates if total_candidates else 0.0
    relative_rows = lookup_rows / active_rows if active_rows else 1.0
    failures: list[str] = []
    if lookup_rows < gates.minimum_lookup_rows:
        failures.append("lookup row count below minimum")
    if relative_rows < gates.minimum_relative_rows:
        failures.append("lookup row count regressed against active dataset")
    if rejected_ratio > gates.maximum_rejected_ratio:
        failures.append("rejected-form ratio above maximum")
    metrics = {
        "lookup_rows": float(lookup_rows),
        "rejected_ratio": rejected_ratio,
        "relative_rows": relative_rows,
    }
    if failures:
        logger.error(
            "quality gates failed",
            extra={"dataset_id": dataset_id, "failures": failures, **metrics},
        )
        raise QualityGateError("; ".join(failures))
    logger.info(
        "quality gates passed", extra={"dataset_id": dataset_id, **metrics}
    )
    return metrics


def publish_dataset(database_url: str, dataset_id: int) -> int | None:
    """Atomically publish a validated dataset and supersede the previous active run."""
    with psycopg.connect(database_url) as connection:
        with connection.transaction():
            row = connection.execute(
                "SELECT status FROM import_run WHERE id = %s FOR UPDATE", (dataset_id,)
            ).fetchone()
            if row is None or row[0] != "validated":
                raise ValueError("dataset must exist in validated state")
            active = connection.execute(
                "SELECT dataset_id FROM active_dataset WHERE singleton FOR UPDATE"
            ).fetchone()
            previous = None if active is None else cast(tuple[int], active)[0]
            if previous is not None and previous != dataset_id:
                connection.execute(
                    """
                    UPDATE import_run SET status = 'superseded'
                    WHERE id = %s AND status = 'published'
                    """,
                    (previous,),
                )
            connection.execute(
                """
                INSERT INTO active_dataset (singleton, dataset_id, activated_at)
                VALUES (TRUE, %s, now())
                ON CONFLICT (singleton) DO UPDATE
                SET dataset_id = EXCLUDED.dataset_id, activated_at = EXCLUDED.activated_at
                """,
                (dataset_id,),
            )
            connection.execute(
                """
                UPDATE import_run
                SET status = 'published', finished_at = COALESCE(finished_at, now())
                WHERE id = %s
                """,
                (dataset_id,),
            )
    logger.info(
        "dataset published",
        extra={"dataset_id": dataset_id, "previous_dataset_id": previous},
    )
    return previous


def rollback_dataset(database_url: str, dataset_id: int) -> int:
    """Atomically activate a retained published or superseded dataset."""
    with psycopg.connect(database_url) as connection:
        with connection.transaction():
            target = connection.execute(
                "SELECT status FROM import_run WHERE id = %s FOR UPDATE", (dataset_id,)
            ).fetchone()
            if target is None or target[0] not in {"published", "superseded"}:
                raise ValueError("rollback target is not a retained published dataset")
            active = connection.execute(
                "SELECT dataset_id FROM active_dataset WHERE singleton FOR UPDATE"
            ).fetchone()
            if active is None:
                raise ValueError("no active dataset to roll back")
            previous = cast(tuple[int], active)[0]
            if previous != dataset_id:
                connection.execute(
                    "UPDATE import_run SET status = 'superseded' WHERE id = %s",
                    (previous,),
                )
                connection.execute(
                    "UPDATE import_run SET status = 'published' WHERE id = %s",
                    (dataset_id,),
                )
                connection.execute(
                    """
                    UPDATE active_dataset
                    SET dataset_id = %s, activated_at = now()
                    WHERE singleton
                    """,
                    (dataset_id,),
                )
    logger.info(
        "dataset rolled back",
        extra={"dataset_id": dataset_id, "previous_dataset_id": previous},
    )
    return previous


def cleanup_retained_datasets(database_url: str, retain_count: int) -> list[int]:
    if retain_count < 0:
        raise ValueError("retain_count cannot be negative")
    with psycopg.connect(database_url) as connection:
        with connection.transaction():
            rows = connection.execute(
                """
                SELECT id
                FROM import_run
                WHERE status IN ('superseded', 'failed')
                  AND id <> (
                      SELECT dataset_id FROM active_dataset WHERE singleton
                  )
                ORDER BY finished_at DESC NULLS LAST, id DESC
                OFFSET %s
                FOR UPDATE
                """,
                (retain_count,),
            ).fetchall()
            identifiers = [cast(tuple[int], row)[0] for row in rows]
            if identifiers:
                connection.execute(
                    "DELETE FROM import_run WHERE id = ANY(%s)", (identifiers,)
                )
    if identifiers:
        logger.info("retained datasets deleted", extra={"dataset_ids": identifiers})
    return identifiers


def import_staging_dataset(
    database_url: str,
    directory: Path,
    *,
    gates: QualityGates | None = None,
) -> tuple[int, dict[str, float]]:
    """Load one staging directory and leave a validated, unpublished dataset."""
    start = time.monotonic()
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    required = (
        "dataset_key",
        "dump_url",
        "dump_sha256",
        "parser_version",
        "normalization_version",
        "schema_version",
    )
    missing = [field for field in required if field not in manifest]
    if missing:
        raise ValueError(f"manifest is missing required fields: {', '.join(missing)}")
    with psycopg.connect(database_url) as connection:
        existing = connection.execute(
            "SELECT id, status FROM import_run WHERE dataset_key = %s",
            (manifest["dataset_key"],),
        ).fetchone()
        if existing is not None:
            dataset_id, status = cast(tuple[int, str], existing)
            if status != "building":
                raise ValueError("dataset key already exists outside a resumable build")
        else:
            row = connection.execute(
                """
                INSERT INTO import_run (
                    dataset_key, dump_url, dump_sha256, dump_timestamp,
                    parser_version, normalization_version, schema_version,
                    git_commit, statistics
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    manifest["dataset_key"],
                    manifest["dump_url"],
                    manifest["dump_sha256"],
                    manifest.get("dump_timestamp"),
                    manifest["parser_version"],
                    manifest["normalization_version"],
                    manifest["schema_version"],
                    manifest.get("git_commit"),
                    Jsonb(manifest.get("statistics", {})),
                ),
            ).fetchone()
            if row is None:
                raise RuntimeError("failed to create import run")
            dataset_id = cast(tuple[int], row)[0]

    logger.info(
        "import started",
        extra={"dataset_id": dataset_id, "dataset_key": manifest["dataset_key"]},
    )
    files = {
        "staging_lexeme": "lexemes.jsonl.zst",
        "staging_word_form": "word_forms.jsonl.zst",
        "staging_stress_variant": "stress_variants.jsonl.zst",
        "staging_source_ref": "source_refs.jsonl.zst",
    }
    copy_started = time.monotonic()
    try:
        for table, filename in files.items():
            with psycopg.connect(database_url) as connection:
                existing_rows = connection.execute(
                    f"SELECT count(*) FROM {table} WHERE dataset_id = %s",  # noqa: S608
                    (dataset_id,),
                ).fetchone()
            staged_count = 0 if existing_rows is None else cast(tuple[int], existing_rows)[0]
            if staged_count:
                continue
            copy_staging_records(
                database_url,
                table,
                dataset_id,
                iter_jsonl_zst(directory / filename),
            )
        copy_duration_seconds = time.monotonic() - copy_started
        with psycopg.connect(database_url) as connection:
            with connection.transaction():
                validate_staging(connection, dataset_id)
                merge_staging(connection, dataset_id)
            # Bulk rows must be committed and analyzed before projection planning.
            # Without fresh statistics PostgreSQL can choose a catastrophic join
            # plan for a newly loaded dataset.
            for table in ("lexeme", "word_form", "stress_variant", "source_ref"):
                connection.execute(
                    sql.SQL("ANALYZE {}").format(sql.Identifier(table))
                )
            connection.commit()
            projection_started = time.monotonic()
            with connection.transaction():
                build_lookup_projection(connection, dataset_id)
                for error in read_jsonl_zst(directory / "parse_errors.jsonl.zst"):
                    connection.execute(
                        """
                        INSERT INTO parse_error (
                            dataset_id, page_id, revision_id, page_title,
                            error_type, error_message, source_fragment
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            dataset_id,
                            error.get("page_id"),
                            error.get("revision_id"),
                            error.get("page_title"),
                            error["error_type"],
                            error["error_message"],
                            error.get("source_fragment", ""),
                        ),
                    )
                unhandled_path = directory / "reports" / "unhandled_templates.json"
                if unhandled_path.exists():
                    for report in json.loads(unhandled_path.read_text(encoding="utf-8")):
                        titles = report.get("sample_page_titles", [""])
                        invocations = report.get("sample_invocations", [""])
                        connection.execute(
                            """
                            INSERT INTO unhandled_template (
                                dataset_id, normalized_name, occurrences,
                                sample_page_title, sample_invocation
                            )
                            VALUES (%s, %s, %s, %s, %s)
                            """,
                            (
                                dataset_id,
                                report["normalized_name"],
                                report["occurrences"],
                                titles[0] if titles else "",
                                invocations[0] if invocations else "",
                            ),
                        )
                metrics = enforce_quality_gates(
                    connection, dataset_id, gates or QualityGates()
                )
                finalize_dataset(connection, dataset_id)
        # projection_duration also includes parse-error/unhandled-template
        # loading and the quality-gate/finalize calls in the same
        # transaction — they cannot be split out without an extra commit,
        # which would change this transaction's atomicity guarantees.
        metrics["copy_duration_seconds"] = round(copy_duration_seconds, 3)
        metrics["projection_duration_seconds"] = round(time.monotonic() - projection_started, 3)
        logger.info(
            "import complete",
            extra={
                "dataset_id": dataset_id,
                "duration_seconds": round(time.monotonic() - start, 3),
                **metrics,
            },
        )
        return dataset_id, metrics
    except Exception:
        logger.exception(
            "import failed",
            extra={
                "dataset_id": dataset_id,
                "duration_seconds": round(time.monotonic() - start, 3),
            },
        )
        with psycopg.connect(database_url) as connection:
            connection.execute(
                """
                UPDATE import_run
                SET status = 'failed', finished_at = now()
                WHERE id = %s AND status = 'building'
                """,
                (dataset_id,),
            )
        raise
