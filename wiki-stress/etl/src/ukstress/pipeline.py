"""End-to-end dump parsing into deterministic staging artifacts."""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any

from ukstress import __version__
from ukstress.deduplicator import lexeme_key, stable_natural_key
from ukstress.downloader import sha256_file
from ukstress.dump_reader import DumpPage, RejectedPage, parse_pages_parallel, stream_pages
from ukstress.entry_parser import redirect_alias
from ukstress.language_sections import ukrainian_entries
from ukstress.morphology import MorphologyCatalog
from ukstress.normalizer import lookup_key, stress_signature
from ukstress.reports import bounded_text, write_quality_reports
from ukstress.staging import (
    ImportManifest,
    read_jsonl_zst,
    write_jsonl_zst_stream,
    write_manifest,
)
from ukstress.table_parser import extract_table_forms
from ukstress.validator import classify_candidate
from ukstress.wikicode import (
    ExtractedCandidate,
    UnhandledTemplateRegistry,
    extract_bold_headwords,
    extract_controlled_fallback,
    extract_templates,
)

STAGING_FILES = {
    "lexemes": "lexemes.jsonl.zst",
    "word_forms": "word_forms.jsonl.zst",
    "stress_variants": "stress_variants.jsonl.zst",
    "source_refs": "source_refs.jsonl.zst",
    "parse_errors": "parse_errors.jsonl.zst",
}

logger = logging.getLogger("ukstress.pipeline")


@dataclass(frozen=True)
class ParsedPage:
    lexemes: tuple[dict[str, Any], ...] = ()
    word_forms: tuple[dict[str, Any], ...] = ()
    stress_variants: tuple[dict[str, Any], ...] = ()
    source_refs: tuple[dict[str, Any], ...] = ()
    errors: tuple[dict[str, Any], ...] = ()
    unhandled: tuple[dict[str, Any], ...] = ()
    aliases: tuple[dict[str, Any], ...] = ()
    accepted: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class ParseSummary:
    dataset_key: str
    output_dir: Path
    statistics: dict[str, int]


class _DiskStagingStore:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._connection = sqlite3.connect(path)
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=NORMAL")
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS staging_record (
                kind TEXT NOT NULL,
                natural_key TEXT NOT NULL,
                payload TEXT NOT NULL,
                PRIMARY KEY (kind, natural_key)
            ) WITHOUT ROWID
            """
        )

    def add(self, kind: str, record: dict[str, Any]) -> bool:
        inserted = self._connection.execute(
            """
            INSERT OR IGNORE INTO staging_record (kind, natural_key, payload)
            VALUES (?, ?, ?)
            """,
            (
                kind,
                str(record["natural_key"]),
                json.dumps(record, ensure_ascii=False, separators=(",", ":"), sort_keys=True),
            ),
        ).rowcount
        return inserted == 0

    def commit(self) -> None:
        self._connection.commit()

    def count(self, kind: str) -> int:
        row = self._connection.execute(
            "SELECT count(*) FROM staging_record WHERE kind = ?", (kind,)
        ).fetchone()
        return 0 if row is None else int(row[0])

    def records(self, kind: str) -> Any:
        rows = self._connection.execute(
            """
            SELECT payload
            FROM staging_record
            WHERE kind = ?
            ORDER BY natural_key
            """,
            (kind,),
        )
        for (payload,) in rows:
            yield json.loads(str(payload))

    def close_and_remove(self) -> None:
        self._connection.close()
        self._path.unlink(missing_ok=True)
        self._path.with_name(self._path.name + "-wal").unlink(missing_ok=True)
        self._path.with_name(self._path.name + "-shm").unlink(missing_ok=True)


def _candidate_records(
    page: DumpPage,
    section: str,
    entry_index: int,
    candidate: ExtractedCandidate,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]] | None:
    validation = classify_candidate(candidate.stressed_form)
    if validation.classification != "valid":
        return None
    lemma_normalized = lookup_key(page.title)
    sense_key = str(entry_index)
    lexeme_natural_key = lexeme_key(
        "staging",
        lemma_normalized,
        candidate.part_of_speech,
        sense_key,
        page.title,
        section,
    )
    morphology_key = "|".join(candidate.grammatical_tags)
    form_normalized = lookup_key(validation.canonical_form)
    word_form_natural_key = stable_natural_key(
        "word_form", lexeme_natural_key, form_normalized, morphology_key
    )
    variant_natural_key = stable_natural_key(
        "stress_variant",
        word_form_natural_key,
        validation.canonical_form,
        stress_signature(validation.canonical_form),
    )
    source_natural_key = stable_natural_key(
        "source_ref",
        lexeme_natural_key,
        word_form_natural_key,
        page.page_id,
        page.revision_id,
        candidate.source_kind,
        validation.canonical_form,
    )
    is_lemma = form_normalized == lemma_normalized
    lexeme = {
        "natural_key": lexeme_natural_key,
        "lemma": page.title,
        "lemma_normalized": lemma_normalized,
        "stressed_lemma": validation.canonical_form if is_lemma else None,
        "part_of_speech": candidate.part_of_speech or "unknown",
        "sense_key": sense_key,
        "source_title": page.title,
        "source_section": section,
        "is_multiword": validation.is_multiword,
        "is_obsolete": False,
        "confidence": candidate.confidence,
    }
    word_form = {
        "natural_key": word_form_natural_key,
        "lexeme_natural_key": lexeme_natural_key,
        "form": validation.canonical_form,
        "form_normalized": form_normalized,
        "grammatical_tags": list(candidate.grammatical_tags),
        "morphology_key": morphology_key,
        "is_lemma": is_lemma,
        "is_variant": False,
        "confidence": candidate.confidence,
        "source_rank": round((1.0 - candidate.confidence) * 100),
    }
    stress_variant = {
        "natural_key": variant_natural_key,
        "word_form_natural_key": word_form_natural_key,
        "stressed_form": validation.canonical_form,
        "stress_signature": stress_signature(validation.canonical_form),
        "variant_type": "primary",
        "confidence": candidate.confidence,
    }
    source_ref = {
        "natural_key": source_natural_key,
        "lexeme_natural_key": lexeme_natural_key,
        "word_form_natural_key": word_form_natural_key,
        "page_id": page.page_id,
        "revision_id": page.revision_id,
        "page_title": page.title,
        "source_kind": candidate.source_kind,
        "source_section": section,
        "source_fragment": bounded_text(candidate.stressed_form),
        "source_offset": None,
    }
    return lexeme, word_form, stress_variant, source_ref


def _parse_page(page: DumpPage, morphology_catalog: MorphologyCatalog) -> ParsedPage:
    alias = redirect_alias(page)
    if alias is not None:
        return ParsedPage(aliases=(asdict(alias),))

    registry = UnhandledTemplateRegistry()
    errors: list[dict[str, Any]] = []
    lexemes: list[dict[str, Any]] = []
    word_forms: list[dict[str, Any]] = []
    stress_variants: list[dict[str, Any]] = []
    source_refs: list[dict[str, Any]] = []
    accepted: list[dict[str, Any]] = []
    entries, unknown_markers = ukrainian_entries(page.wikitext)
    for marker in unknown_markers:
        errors.append(
            {
                "natural_key": stable_natural_key(
                    "parse_error", page.page_id, page.revision_id, "unknown_language_marker", marker
                ),
                "page_id": page.page_id,
                "revision_id": page.revision_id,
                "page_title": page.title,
                "error_type": "unknown_language_marker",
                "error_message": bounded_text(marker),
                "source_fragment": bounded_text(marker),
            }
        )
    for entry_index, entry in enumerate(entries):
        template_candidates, _ = extract_templates(
            entry.content,
            page_title=page.title,
            unhandled_registry=registry,
            morphology_catalog=morphology_catalog,
        )
        table_candidates = [
            ExtractedCandidate(form.stressed_form, "manual_table", None, 0.75)
            for form in extract_table_forms(entry.content)
        ]
        candidates = template_candidates + table_candidates
        if not candidates:
            candidates = extract_bold_headwords(entry.content)
        if not candidates:
            candidates = extract_controlled_fallback(entry.content)
        for candidate in candidates:
            records = _candidate_records(page, entry.heading, entry_index, candidate)
            if records is None:
                validation = classify_candidate(candidate.stressed_form)
                errors.append(
                    {
                        "natural_key": stable_natural_key(
                            "parse_error",
                            page.page_id,
                            page.revision_id,
                            "rejected_form",
                            candidate.stressed_form,
                        ),
                        "page_id": page.page_id,
                        "revision_id": page.revision_id,
                        "page_title": page.title,
                        "error_type": "rejected_form",
                        "error_message": "; ".join(validation.reasons),
                        "source_fragment": bounded_text(candidate.stressed_form),
                    }
                )
                continue
            lexeme, word_form, stress_variant, source_ref = records
            lexemes.append(lexeme)
            word_forms.append(word_form)
            stress_variants.append(stress_variant)
            source_refs.append(source_ref)
            accepted.append(
                {
                    "part_of_speech": candidate.part_of_speech or "unknown",
                    "source_kind": candidate.source_kind,
                    "confidence": candidate.confidence,
                    "grammatical_tags": list(candidate.grammatical_tags),
                }
            )
    return ParsedPage(
        lexemes=tuple(lexemes),
        word_forms=tuple(word_forms),
        stress_variants=tuple(stress_variants),
        source_refs=tuple(source_refs),
        errors=tuple(errors),
        unhandled=tuple(asdict(report) for report in registry.reports()),
        accepted=tuple(accepted),
    )


def _deduplicate(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    unique: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []
    for record in records:
        key = str(record["natural_key"])
        if key in unique:
            duplicates.append(key)
            existing = unique[key]
            if "confidence" in record:
                existing["confidence"] = max(
                    float(existing["confidence"]), float(record["confidence"])
                )
        else:
            unique[key] = dict(record)
    return list(unique.values()), duplicates


def parse_dump(
    dump_path: Path,
    output_root: Path,
    *,
    workers: int = 1,
    dataset_key: str | None = None,
    dump_url: str = "",
) -> ParseSummary:
    """Stream a dump and write one complete deterministic staging directory."""
    start = time.monotonic()
    dump_sha256 = sha256_file(dump_path)
    resolved_key = dataset_key or dump_sha256[:16]
    output_dir = output_root / resolved_key
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "parse started",
        extra={"dataset_key": resolved_key, "dump_path": str(dump_path), "workers": workers},
    )
    rejected_pages: list[RejectedPage] = []
    morphology_catalog = MorphologyCatalog.from_dump(dump_path)
    pages = stream_pages(dump_path, on_rejected=rejected_pages.append)
    parsed_pages = parse_pages_parallel(
        pages, workers, partial(_parse_page, morphology_catalog=morphology_catalog)
    )
    store = _DiskStagingStore(output_dir / ".parse-staging.sqlite")
    aliases: list[dict[str, Any]] = []
    accepted_counts = {
        "part_of_speech": Counter[str](),
        "source_kind": Counter[str](),
        "confidence": Counter[str](),
        "grammatical_feature": Counter[str](),
    }
    issue_errors: list[dict[str, Any]] = []
    duplicate_keys: list[str] = []
    unhandled_counts: Counter[str] = Counter()
    unhandled_samples: dict[str, dict[str, Any]] = {}
    page_count = 0
    for parsed in parsed_pages:
        page_count += 1
        for name, values in (
            ("lexemes", parsed.lexemes),
            ("word_forms", parsed.word_forms),
            ("stress_variants", parsed.stress_variants),
            ("source_refs", parsed.source_refs),
            ("parse_errors", parsed.errors),
        ):
            for value in values:
                if store.add(name, value):
                    duplicate_keys.append(str(value["natural_key"]))
                if name == "parse_errors":
                    issue_errors.append(value)
        aliases.extend(parsed.aliases)
        for row in parsed.accepted:
            accepted_counts["part_of_speech"][
                str(row.get("part_of_speech", "unknown"))
            ] += 1
            accepted_counts["source_kind"][str(row.get("source_kind", "unknown"))] += 1
            accepted_counts["confidence"][
                f"{float(row.get('confidence', 0.0)):.2f}"
            ] += 1
            accepted_counts["grammatical_feature"].update(
                str(tag) for tag in row.get("grammatical_tags", [])
            )
        for report in parsed.unhandled:
            name = str(report["normalized_name"])
            unhandled_counts[name] += int(report["occurrences"])
            unhandled_samples.setdefault(name, report)
        if page_count % 250 == 0:
            store.commit()
    for rejected in rejected_pages:
        error = {
                "natural_key": stable_natural_key(
                    "parse_error", rejected.title, rejected.reason
                ),
                "page_id": None,
                "revision_id": None,
                "page_title": rejected.title,
                "error_type": "rejected_page",
                "error_message": rejected.reason,
                "source_fragment": "",
            }
        if store.add("parse_errors", error):
            duplicate_keys.append(str(error["natural_key"]))
        issue_errors.append(error)

    store.commit()
    statistics: dict[str, int] = {"pages": page_count, "aliases": len(aliases)}
    for name, filename in STAGING_FILES.items():
        statistics[name] = write_jsonl_zst_stream(
            output_dir / filename, store.records(name)
        )

    unhandled = []
    for name in sorted(unhandled_counts):
        row = dict(unhandled_samples[name])
        row["occurrences"] = unhandled_counts[name]
        unhandled.append(row)
    issues = {
        "parse_errors": issue_errors,
        "rejected_forms": [
            row for row in issue_errors if row["error_type"] == "rejected_form"
        ],
        "unhandled_templates": unhandled,
        "duplicates": [{"natural_key": key} for key in sorted(set(duplicate_keys))],
        "ambiguous_forms": [],
    }
    write_quality_reports(
        output_dir / "reports",
        issues=issues,
        accepted=[],
        aggregate_counts=accepted_counts,
    )
    (output_dir / "aliases.json").write_text(
        json.dumps(aliases, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    now = datetime.now(UTC).isoformat()
    write_manifest(
        output_dir / "manifest.json",
        ImportManifest(
            dataset_key=resolved_key,
            dump_url=dump_url,
            dump_sha256=dump_sha256,
            dump_timestamp=None,
            parser_version=__version__,
            normalization_version="1",
            schema_version="006",
            git_commit=None,
            command_options={"workers": workers},
            started_at=now,
            finished_at=now,
            statistics=statistics,
        ),
    )
    store.close_and_remove()
    duration_seconds = time.monotonic() - start
    logger.info(
        "parse complete",
        extra={
            "dataset_key": resolved_key,
            "duration_seconds": round(duration_seconds, 3),
            "pages_per_second": round(page_count / duration_seconds, 2)
            if duration_seconds > 0
            else None,
            "unhandled_template_count": len(unhandled_counts),
            "duplicate_count": len(duplicate_keys),
            "error_count": len(issue_errors),
            **statistics,
        },
    )
    return ParseSummary(resolved_key, output_dir, statistics)


def validate_staging_directory(directory: Path) -> dict[str, int]:
    """Validate required staging files and all natural-key relationships offline."""
    records = {
        name: read_jsonl_zst(directory / filename)
        for name, filename in STAGING_FILES.items()
    }
    lexeme_keys = {str(row["natural_key"]) for row in records["lexemes"]}
    word_form_keys = {str(row["natural_key"]) for row in records["word_forms"]}
    for row in records["word_forms"]:
        if str(row["lexeme_natural_key"]) not in lexeme_keys:
            raise ValueError("word form references an unknown lexeme")
    for row in records["stress_variants"]:
        if str(row["word_form_natural_key"]) not in word_form_keys:
            raise ValueError("stress variant references an unknown word form")
    for row in records["source_refs"]:
        lexeme_key_value = row.get("lexeme_natural_key")
        word_form_key_value = row.get("word_form_natural_key")
        if lexeme_key_value is not None and str(lexeme_key_value) not in lexeme_keys:
            raise ValueError("source reference references an unknown lexeme")
        if word_form_key_value is not None and str(word_form_key_value) not in word_form_keys:
            raise ValueError("source reference references an unknown word form")
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("dataset_key") != directory.name:
        raise ValueError("manifest dataset key does not match staging directory")
    return {name: len(values) for name, values in records.items()}
