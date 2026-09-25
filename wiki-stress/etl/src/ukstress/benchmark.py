"""ETL benchmark: measures a real parse + import run and reports actual
numbers. Every figure here comes from timing/measuring this run; nothing
is a pre-populated target (openspec/config.yaml's context.rules: "All
performance claims must be measured, not invented.").
"""

from __future__ import annotations

import platform
import resource
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import psycopg

from ukstress.database import QualityGates, apply_migrations, import_staging_dataset
from ukstress.pipeline import parse_dump


@dataclass(frozen=True)
class BenchmarkReport:
    generated_at: str
    hardware: dict[str, str]
    dump_path: str
    dump_byte_size: int
    workers: int
    parse: dict[str, Any]
    staging: dict[str, Any]
    importing: dict[str, Any]
    tables: dict[str, Any]


def _peak_rss_megabytes() -> float:
    """Peak resident set size for this process, in megabytes.

    ru_maxrss is kilobytes on Linux and bytes on macOS (Darwin); both
    platforms report it in the platform's own native unit, not a
    documented cross-platform one.
    """
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1024.0 if platform.system() == "Darwin" else 1.0
    return round(raw / divisor / 1024.0, 1)


def _directory_size_bytes(directory: Path) -> int:
    return sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())


def _table_and_index_sizes(database_url: str, dataset_id: int) -> dict[str, Any]:
    # Table names are relation names for pg_*_size(), which take a
    # regclass argument bound as a normal parameter (%s) — not string
    # interpolation into the query text — so this is not a SQL-injection
    # surface despite the table name flowing into a query string below for
    # the row-count select. That select's table name comes only from this
    # fixed tuple, never from user or dump input.
    tables = ("lexeme", "word_form", "stress_variant", "source_ref", "stress_lookup")
    sizes: dict[str, Any] = {}
    with psycopg.connect(database_url) as connection:
        for table in tables:
            total_bytes, table_bytes, index_bytes = connection.execute(
                "SELECT pg_total_relation_size(%s), pg_relation_size(%s), pg_indexes_size(%s)",
                (table, table, table),
            ).fetchone()  # type: ignore[misc]
            row_count = connection.execute(
                f"SELECT count(*) FROM {table} WHERE dataset_id = %s",  # noqa: S608
                (dataset_id,),
            ).fetchone()
            sizes[table] = {
                "row_count": row_count[0] if row_count else 0,
                "total_bytes": total_bytes,
                "table_bytes": table_bytes,
                "index_bytes": index_bytes,
            }
    return sizes


def run_benchmark(
    dump_path: Path,
    output_root: Path,
    database_url: str,
    migrations_dir: Path,
    *,
    workers: int = 1,
) -> BenchmarkReport:
    apply_migrations(database_url, migrations_dir)

    parse_start = time.monotonic()
    summary = parse_dump(dump_path, output_root, workers=workers, dump_url="")
    parse_duration = time.monotonic() - parse_start
    peak_rss_mb = _peak_rss_megabytes()

    pages = summary.statistics.get("pages", 0)
    word_forms = summary.statistics.get("word_forms", 0)
    staging_bytes = _directory_size_bytes(summary.output_dir)

    dataset_id, import_metrics = import_staging_dataset(
        database_url,
        summary.output_dir,
        gates=QualityGates(
            minimum_lookup_rows=0, minimum_relative_rows=0.0, maximum_rejected_ratio=1.0
        ),
    )

    tables = _table_and_index_sizes(database_url, dataset_id)

    return BenchmarkReport(
        generated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        hardware={
            "platform": platform.platform(),
            "processor": platform.processor() or platform.machine(),
            "python_version": platform.python_version(),
        },
        dump_path=str(dump_path),
        dump_byte_size=dump_path.stat().st_size,
        workers=workers,
        parse={
            "duration_seconds": round(parse_duration, 3),
            "pages": pages,
            "pages_per_second": round(pages / parse_duration, 2) if parse_duration > 0 else None,
            "word_forms": word_forms,
            "word_forms_per_second": round(word_forms / parse_duration, 2)
            if parse_duration > 0
            else None,
            "peak_rss_megabytes": peak_rss_mb,
        },
        staging={
            "output_dir": str(summary.output_dir),
            "total_bytes": staging_bytes,
        },
        importing={
            "dataset_id": dataset_id,
            "copy_duration_seconds": import_metrics.get("copy_duration_seconds"),
            "projection_duration_seconds": import_metrics.get("projection_duration_seconds"),
            "lookup_rows": import_metrics.get("lookup_rows"),
        },
        tables=tables,
    )


def report_dict(report: BenchmarkReport) -> dict[str, Any]:
    return asdict(report)


def render_markdown(report: BenchmarkReport) -> str:
    lines = [
        "# ETL benchmark report",
        "",
        f"Generated at {report.generated_at}. Dump: `{report.dump_path}` "
        f"({report.dump_byte_size:,} bytes), {report.workers} worker(s).",
        "",
        f"- Hardware: {report.hardware['platform']}, {report.hardware['processor']}, "
        f"Python {report.hardware['python_version']}",
        "",
        "## Parse",
        "",
        f"- Duration: {report.parse['duration_seconds']} s",
        f"- Pages: {report.parse['pages']} ({report.parse['pages_per_second']} pages/s)",
        f"- Word forms: {report.parse['word_forms']} "
        f"({report.parse['word_forms_per_second']} forms/s)",
        f"- Peak RSS: {report.parse['peak_rss_megabytes']} MB",
        "",
        "## Staging",
        "",
        f"- Output directory: `{report.staging['output_dir']}`",
        f"- Total size: {report.staging['total_bytes']:,} bytes",
        "",
        "## Import",
        "",
        f"- Dataset ID: {report.importing['dataset_id']}",
        f"- COPY duration: {report.importing['copy_duration_seconds']} s",
        f"- Projection duration: {report.importing['projection_duration_seconds']} s",
        f"- Lookup rows: {report.importing['lookup_rows']}",
        "",
        "## Table and index sizes",
        "",
        "| Table | Rows | Total | Table | Indexes |",
        "| --- | --- | --- | --- | --- |",
    ]
    for table, sizes in report.tables.items():
        lines.append(
            f"| {table} | {sizes['row_count']:,} | {sizes['total_bytes']:,} B | "
            f"{sizes['table_bytes']:,} B | {sizes['index_bytes']:,} B |"
        )
    lines.append("")
    return "\n".join(lines)
