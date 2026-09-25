"""Merge an accented wordlist into an existing staging directory.

The import contract is dataset-scoped: `stress_lookup` is rebuilt per dataset,
so a dataset must be complete rather than incremental. This module therefore
emits a *new* staging directory holding the primary source's records plus the
wordlist forms that source never covered.

Record volume is ~9M, which is too much to sort in memory, so records are
bucketed by natural-key prefix on disk and each bucket is sorted independently.
That keeps peak memory proportional to one bucket while still handing
`write_jsonl_zst_stream` the strictly increasing keys it requires.
"""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ukstress.normalizer import stress_signature
from ukstress.sources.stressed_wordlist import (
    WordlistStats,
    audit_wordlist,
    build_form_records,
    iter_wordlist_forms,
)
from ukstress.staging import ImportManifest, iter_jsonl_zst, write_jsonl_zst_stream, write_manifest

logger = logging.getLogger(__name__)

_BUCKETS = 256
_RECORD_KINDS = ("lexemes", "word_forms", "stress_variants", "source_refs")


class _BucketedWriter:
    """Collect records on disk, then emit them ordered by natural key."""

    def __init__(self, directory: Path, name: str) -> None:
        self._directory = directory / name
        self._directory.mkdir(parents=True, exist_ok=True)
        self._handles = {
            index: (self._directory / f"{index:02x}.jsonl").open("w", encoding="utf-8")
            for index in range(_BUCKETS)
        }
        self.count = 0

    def add(self, record: dict[str, Any]) -> None:
        key = str(record["natural_key"])
        handle = self._handles[int(key[:2], 16)]
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        self.count += 1

    def _iter_sorted(self) -> Iterator[dict[str, Any]]:
        seen_previous: str | None = None
        for index in range(_BUCKETS):
            path = self._directory / f"{index:02x}.jsonl"
            records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            records.sort(key=lambda record: str(record["natural_key"]))
            for record in records:
                key = str(record["natural_key"])
                if key == seen_previous:
                    # Same identity emitted by both sources; the contract wants
                    # one row, and the payloads are identical by construction.
                    continue
                seen_previous = key
                yield record
            path.unlink()

    def finalize(self, path: Path) -> int:
        for handle in self._handles.values():
            handle.close()
        written = write_jsonl_zst_stream(path, self._iter_sorted())
        shutil.rmtree(self._directory, ignore_errors=True)
        return written


#: Files the importer reads besides the four record streams and the manifest.
#: `parse_errors.jsonl.zst` is required rather than optional
#: (`database.import_staging_dataset`), so a merged directory that omits it
#: fails only at the very end of a multi-hour import.
REQUIRED_AUXILIARY_FILES = ("parse_errors.jsonl.zst",)
OPTIONAL_AUXILIARY_FILES = ("aliases.json",)
CARRIED_REPORTS = (
    "unhandled_templates.json",
    "rejected_forms.json",
    "ambiguous_forms.json",
    "parse_errors.json",
    "counts.json",
    "duplicates.json",
)


def _carry_forward_auxiliary_files(base_dir: Path, output_dir: Path) -> None:
    """Copy the primary source's provenance files into the merged dataset.

    The merged dataset still contains every record the primary parse produced,
    so its parse errors and unhandled-template counts describe it too. Dropping
    them would silently reset the coverage-gap reporting to zero.
    """
    for name in REQUIRED_AUXILIARY_FILES:
        source = base_dir / name
        if not source.exists():
            raise FileNotFoundError(
                f"primary staging directory is missing {name}, which the importer requires"
            )
        shutil.copy2(source, output_dir / name)
    for name in OPTIONAL_AUXILIARY_FILES:
        source = base_dir / name
        if source.exists():
            shutil.copy2(source, output_dir / name)
    for name in CARRIED_REPORTS:
        source = base_dir / "reports" / name
        if source.exists():
            shutil.copy2(source, output_dir / "reports" / name)


def _primary_signatures(base_dir: Path) -> dict[str, set[str]]:
    """Map every form the primary source covers to its stress signatures."""
    form_by_key: dict[str, str] = {}
    for record in iter_jsonl_zst(base_dir / "word_forms.jsonl.zst"):
        form_by_key[str(record["natural_key"])] = str(record["form_normalized"])
    signatures: dict[str, set[str]] = {}
    for record in iter_jsonl_zst(base_dir / "stress_variants.jsonl.zst"):
        form = form_by_key.get(str(record["word_form_natural_key"]))
        if form is None:
            continue
        signatures.setdefault(form, set()).add(str(record["stress_signature"]))
    # Forms present but carrying no variant still belong to the primary source.
    for form in form_by_key.values():
        signatures.setdefault(form, set())
    return signatures


def merge_wordlist_into_staging(
    base_dir: Path,
    wordlist_path: Path,
    output_dir: Path,
    *,
    dataset_key: str,
    wordlist_sha256: str,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write a merged staging directory and return its statistics."""
    started_at = datetime.now(UTC).isoformat()
    audit = audit_wordlist(wordlist_path)
    if audit["multi_token"] or audit["hyphenated"]:
        raise ValueError(
            "wordlist contains multi-token or hyphenated entries; alternative-stress "
            "splitting is only valid for single tokens "
            f"(multi_token={audit['multi_token']}, hyphenated={audit['hyphenated']})"
        )

    logger.info("loading primary source forms", extra={"base_dir": str(base_dir)})
    primary = _primary_signatures(base_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "reports").mkdir(parents=True, exist_ok=True)
    _carry_forward_auxiliary_files(base_dir, output_dir)
    conflict_path = output_dir / "reports" / "stress_source_conflicts.tsv"

    stats = WordlistStats()
    with tempfile.TemporaryDirectory(prefix="ukstress-merge-") as scratch:
        scratch_dir = Path(scratch)
        writers = {kind: _BucketedWriter(scratch_dir, kind) for kind in _RECORD_KINDS}

        logger.info("copying primary staging records")
        base_counts = {}
        for kind in _RECORD_KINDS:
            count = 0
            for record in iter_jsonl_zst(base_dir / f"{kind}.jsonl.zst"):
                writers[kind].add(record)
                count += 1
            base_counts[kind] = count

        logger.info("merging wordlist forms", extra={"wordlist": str(wordlist_path)})
        with conflict_path.open("w", encoding="utf-8") as conflicts:
            conflicts.write("form_normalized\tprimary_signatures\twordlist_signatures\tresolution\n")
            for form in iter_wordlist_forms(wordlist_path, stats):
                wordlist_signatures = {
                    stress_signature(value) for value in form.stressed_forms
                }
                existing = primary.get(form.form_normalized)
                if existing is not None:
                    stats.skipped_present_in_primary += 1
                    if existing and existing != wordlist_signatures:
                        stats.conflicts += 1
                        conflicts.write(
                            f"{form.form_normalized}\t"
                            f"{'|'.join(sorted(existing))}\t"
                            f"{'|'.join(sorted(wordlist_signatures))}\t"
                            "primary_wins\n"
                        )
                    elif existing:
                        stats.agreements += 1
                    continue

                stats.accepted_forms += 1
                if len(wordlist_signatures) > 1:
                    stats.ambiguous_forms += 1
                lexeme, word_form, variants, source_ref = build_form_records(
                    form, dataset_key=dataset_key
                )
                writers["lexemes"].add(lexeme)
                writers["word_forms"].add(word_form)
                for variant in variants:
                    writers["stress_variants"].add(variant)
                writers["source_refs"].add(source_ref)

        logger.info("writing merged staging artifacts", extra={"output_dir": str(output_dir)})
        written = {
            kind: writers[kind].finalize(output_dir / f"{kind}.jsonl.zst")
            for kind in _RECORD_KINDS
        }

    base_manifest = json.loads((base_dir / "manifest.json").read_text(encoding="utf-8"))
    statistics = {
        "lexemes": written["lexemes"],
        "word_forms": written["word_forms"],
        "stress_variants": written["stress_variants"],
        "source_refs": written["source_refs"],
        "primary_lexemes": base_counts["lexemes"],
        "primary_word_forms": base_counts["word_forms"],
        **stats.as_dict(),
    }
    write_manifest(
        output_dir / "manifest.json",
        ImportManifest(
            dataset_key=dataset_key,
            dump_url=str(base_manifest.get("dump_url", "")),
            dump_sha256=str(base_manifest.get("dump_sha256", "")),
            dump_timestamp=base_manifest.get("dump_timestamp"),
            parser_version=str(base_manifest.get("parser_version", "")),
            normalization_version=str(base_manifest.get("normalization_version", "")),
            schema_version=str(base_manifest.get("schema_version", "")),
            git_commit=base_manifest.get("git_commit"),
            command_options={
                "merge_source": "stressed_wordlist",
                "primary_dataset_key": base_manifest.get("dataset_key"),
                "wordlist_path": wordlist_path.name,
                "wordlist_sha256": wordlist_sha256,
                "wordlist_audit": audit,
                "conflict_policy": "primary_wins",
                **(provenance or {}),
            },
            started_at=started_at,
            finished_at=datetime.now(UTC).isoformat(),
            statistics=statistics,
        ),
    )
    (output_dir / "reports" / "wordlist_merge.json").write_text(
        json.dumps(statistics, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return statistics
