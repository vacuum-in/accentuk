"""Command-line entry point for the Python-owned ETL lifecycle."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import psycopg

from ukstress.benchmark import render_markdown, report_dict, run_benchmark
from ukstress.database import (
    QualityGates,
    apply_migrations,
    cleanup_retained_datasets,
    import_staging_dataset,
    publish_dataset,
    rollback_dataset,
)
from ukstress.downloader import DumpManifest, download_dump, sha256_file
from ukstress.exports import (
    database_statistics,
    export_rows,
    unhandled_template_report,
    write_export,
)
from ukstress.lifecycle import build_dataset
from ukstress.logging_config import configure_logging
from ukstress.pipeline import parse_dump, validate_staging_directory
from ukstress.serving_inventory import import_serving_inventory
from ukstress.serving_check import check_and_fix
from ukstress.sources.merge import merge_wordlist_into_staging

DEFAULT_DUMP_URL = (
    "https://dumps.wikimedia.org/ukwiktionary/latest/"
    "ukwiktionary-latest-pages-articles.xml.bz2"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ukstress")
    subcommands = parser.add_subparsers(dest="command", required=True)

    download = subcommands.add_parser("download", help="download and verify a Wiktionary dump")
    download.add_argument("--url", default=DEFAULT_DUMP_URL)
    download.add_argument(
        "--output",
        type=Path,
        default=Path("data/ukwiktionary-latest-pages-articles.xml.bz2"),
    )
    download.add_argument("--sha256")
    download.add_argument("--retries", type=int, default=0)

    parse = subcommands.add_parser("parse", help="parse a dump into staging artifacts")
    parse.add_argument("--dump", type=Path, required=True)
    parse.add_argument("--output", type=Path, default=Path("output"))
    parse.add_argument("--dataset-key")
    parse.add_argument("--workers", type=int, default=1)
    parse.add_argument("--dump-url", default="")

    validate = subcommands.add_parser("validate", help="validate staging artifacts offline")
    validate.add_argument("--input", type=Path, required=True)

    migrate = subcommands.add_parser("migrate", help="apply forward-only migrations")
    migrate.add_argument("--database-url", required=True)
    migrate.add_argument("--migrations", type=Path, required=True)
    migrate.add_argument("--check", action="store_true")

    import_command = subcommands.add_parser(
        "import", help="load staging artifacts as a validated dataset"
    )
    import_command.add_argument("--database-url", required=True)
    import_command.add_argument("--input", type=Path, required=True)
    import_command.add_argument("--minimum-lookup-rows", type=int, default=1)
    import_command.add_argument("--minimum-relative-rows", type=float, default=0.8)
    import_command.add_argument("--maximum-rejected-ratio", type=float, default=0.2)

    merge_wordlist = subcommands.add_parser(
        "merge-wordlist",
        help="merge an accented wordlist into a staging directory as a secondary source",
    )
    merge_wordlist.add_argument("--input", type=Path, required=True)
    merge_wordlist.add_argument("--wordlist", type=Path, required=True)
    merge_wordlist.add_argument("--output", type=Path, required=True)
    merge_wordlist.add_argument("--dataset-key", required=True)

    serving_inventory = subcommands.add_parser(
        "import-serving-inventory",
        help="load a frozen contextual-model candidate manifest",
    )
    serving_inventory.add_argument("--database-url", required=True)
    serving_inventory.add_argument("--manifest", type=Path, required=True)

    serving_check = subcommands.add_parser(
        "check-serving",
        help="check that the manifest, the database and the lexicon agree on what the model serves",
    )
    serving_check.add_argument("--database-url", required=True)
    serving_check.add_argument("--manifest", type=Path, required=True,
                               help="the manifest the model service loads")
    serving_check.add_argument("--api-manifest", type=Path,
                               help="the manifest the API loads, if it is a different file")
    serving_check.add_argument("--active-hash", required=True, help="ACTIVE_INVENTORY_HASH")
    serving_check.add_argument("--supplementary", default="", help="SUPPLEMENTARY_DATASETS")
    serving_check.add_argument("--exclusive", default="13", help="REVIEWED_EXCLUSIVE_DATASETS")
    serving_check.add_argument("--token-model-dir", type=Path)
    serving_check.add_argument("--fix", action="store_true",
                               help="write a repaired manifest (backed up first) and reload the database")

    publish = subcommands.add_parser("publish", help="atomically publish a dataset")
    publish.add_argument("--database-url", required=True)
    publish.add_argument("dataset_id", type=int)

    rollback = subcommands.add_parser("rollback", help="atomically activate a retained dataset")
    rollback.add_argument("--database-url", required=True)
    rollback.add_argument("dataset_id", type=int)

    cleanup = subcommands.add_parser(
        "cleanup", help="delete superseded/failed datasets beyond the retention count"
    )
    cleanup.add_argument("--database-url", required=True)
    cleanup.add_argument("--retain-count", type=int, default=1)

    build = subcommands.add_parser("build", help="migrate, parse, validate, and import")
    build.add_argument("--dump", type=Path, required=True)
    build.add_argument("--output", type=Path, default=Path("output"))
    build.add_argument("--database-url", required=True)
    build.add_argument("--migrations", type=Path, required=True)
    build.add_argument("--dataset-key")
    build.add_argument("--workers", type=int, default=1)
    build.add_argument("--publish", action="store_true")

    stats = subcommands.add_parser("stats", help="show deterministic dataset statistics")
    stats.add_argument("--database-url", required=True)
    stats.add_argument("--dataset-id", type=int)

    unhandled = subcommands.add_parser(
        "report-unhandled", help="show unsupported template counts and samples"
    )
    unhandled.add_argument("--database-url", required=True)
    unhandled.add_argument("--dataset-id", type=int)

    export = subcommands.add_parser("export", help="export the read projection")
    export.add_argument("--database-url", required=True)
    export.add_argument("--dataset-id", type=int)
    export.add_argument("--format", choices=("tsv", "jsonl", "tts"), required=True)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--best-only", action="store_true")

    benchmark = subcommands.add_parser(
        "benchmark", help="run and measure a real parse + import, write a machine-readable report"
    )
    benchmark.add_argument("--dump", type=Path, required=True)
    benchmark.add_argument("--output", type=Path, default=Path("output"))
    benchmark.add_argument("--database-url", required=True)
    benchmark.add_argument("--migrations", type=Path, required=True)
    benchmark.add_argument("--workers", type=int, default=1)
    benchmark.add_argument("--report", type=Path, default=Path("reports/etl_benchmark.json"))
    benchmark.add_argument(
        "--report-markdown", type=Path, default=Path("reports/etl_benchmark.md")
    )
    return parser


def _download(arguments: argparse.Namespace) -> DumpManifest:
    if arguments.retries < 0:
        raise ValueError("retries cannot be negative")
    attempts = arguments.retries + 1
    for attempt in range(attempts):
        try:
            return download_dump(arguments.url, arguments.output, arguments.sha256)
        except OSError:
            if attempt + 1 == attempts:
                raise
    raise AssertionError("download retry loop exhausted")


def main(argv: Sequence[str] | None = None) -> int:
    configure_logging()
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "download":
            manifest = _download(arguments)
            print(json.dumps(asdict(manifest), ensure_ascii=False, sort_keys=True))
        elif arguments.command == "parse":
            summary = parse_dump(
                arguments.dump,
                arguments.output,
                workers=arguments.workers,
                dataset_key=arguments.dataset_key,
                dump_url=arguments.dump_url,
            )
            print(
                json.dumps(
                    {
                        "dataset_key": summary.dataset_key,
                        "output_dir": str(summary.output_dir),
                        "statistics": summary.statistics,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
        elif arguments.command == "validate":
            counts = validate_staging_directory(arguments.input)
            print(json.dumps({"valid": True, "counts": counts}, sort_keys=True))
        elif arguments.command == "migrate":
            applied = apply_migrations(
                arguments.database_url, arguments.migrations, check_only=arguments.check
            )
            print(json.dumps({"pending" if arguments.check else "applied": applied}))
        elif arguments.command == "import":
            dataset_id, metrics = import_staging_dataset(
                arguments.database_url,
                arguments.input,
                gates=QualityGates(
                    minimum_lookup_rows=arguments.minimum_lookup_rows,
                    minimum_relative_rows=arguments.minimum_relative_rows,
                    maximum_rejected_ratio=arguments.maximum_rejected_ratio,
                ),
            )
            print(
                json.dumps(
                    {"dataset_id": dataset_id, "status": "validated", "quality": metrics},
                    sort_keys=True,
                )
            )
        elif arguments.command == "merge-wordlist":
            merge_result = merge_wordlist_into_staging(
                arguments.input,
                arguments.wordlist,
                arguments.output,
                dataset_key=arguments.dataset_key,
                wordlist_sha256=sha256_file(arguments.wordlist),
            )
            print(json.dumps(merge_result, ensure_ascii=False, sort_keys=True))
        elif arguments.command == "import-serving-inventory":
            result = import_serving_inventory(arguments.database_url, arguments.manifest)
            print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        elif arguments.command == "check-serving":
            ids = lambda value: [int(x) for x in value.split(",") if x.strip()]  # noqa: E731
            result = check_and_fix(
                arguments.database_url, manifest_path=arguments.manifest, fix=arguments.fix,
                api_manifest_path=arguments.api_manifest, active_hash=arguments.active_hash,
                supplementary=ids(arguments.supplementary), exclusive=ids(arguments.exclusive),
                token_model_dir=arguments.token_model_dir)
            print(json.dumps(result, ensure_ascii=False, indent=1, sort_keys=True))
            if not result.get("ok_after_fix", result["ok"]):
                return 1
        elif arguments.command == "publish":
            previous = publish_dataset(arguments.database_url, arguments.dataset_id)
            print(json.dumps({"published": arguments.dataset_id, "previous": previous}))
        elif arguments.command == "rollback":
            previous = rollback_dataset(arguments.database_url, arguments.dataset_id)
            print(json.dumps({"active": arguments.dataset_id, "previous": previous}))
        elif arguments.command == "cleanup":
            deleted = cleanup_retained_datasets(
                arguments.database_url, arguments.retain_count
            )
            print(json.dumps({"deleted": deleted, "retain_count": arguments.retain_count}))
        elif arguments.command == "build":
            result = build_dataset(
                dump_path=arguments.dump,
                output_root=arguments.output,
                database_url=arguments.database_url,
                migrations_dir=arguments.migrations,
                workers=arguments.workers,
                dataset_key=arguments.dataset_key,
                publish=arguments.publish,
            )
            print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        elif arguments.command == "stats":
            stats_result = database_statistics(
                arguments.database_url, arguments.dataset_id
            )
            print(json.dumps(stats_result, ensure_ascii=False, sort_keys=True))
        elif arguments.command == "report-unhandled":
            unhandled_result = unhandled_template_report(
                arguments.database_url, arguments.dataset_id
            )
            print(json.dumps(unhandled_result, ensure_ascii=False, sort_keys=True))
        elif arguments.command == "export":
            rows = export_rows(
                arguments.database_url,
                dataset_id=arguments.dataset_id,
                best_only=arguments.best_only,
            )
            count = write_export(arguments.output, rows, arguments.format)
            print(json.dumps({"exported": count, "output": str(arguments.output)}))
        elif arguments.command == "benchmark":
            report = run_benchmark(
                arguments.dump,
                arguments.output,
                arguments.database_url,
                arguments.migrations,
                workers=arguments.workers,
            )
            arguments.report.parent.mkdir(parents=True, exist_ok=True)
            payload = report_dict(report)
            arguments.report.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            arguments.report_markdown.parent.mkdir(parents=True, exist_ok=True)
            arguments.report_markdown.write_text(render_markdown(report), encoding="utf-8")
            print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        else:
            raise AssertionError(f"unhandled command: {arguments.command}")
    except KeyboardInterrupt:
        print("ukstress: interrupted", file=sys.stderr)
        return 130
    except psycopg.Error as error:
        # str(error) is libpq/psycopg's own message text, which never
        # embeds the connecting password; letting this fall through to
        # the default uncaught-exception path would instead print a full
        # Python traceback, which is unnecessary noise for a CLI failure.
        print(f"ukstress: database error: {error}", file=sys.stderr)
        return 1
    except (OSError, RuntimeError, ValueError) as error:
        print(f"ukstress: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
