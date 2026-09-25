"""Deterministic, license-aware export of accepted stress-mining records."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from ukstress.datasets import MiningRecord
from ukstress.datasets.models import SCHEMA_VERSION
from ukstress.provenance.fingerprints import content_fingerprint
from ukstress.provenance.governance import ExportLicensePolicy


@dataclass(frozen=True)
class TextResolverExportPolicy:
    include_audio_reference: bool = False
    require_redistribution: bool = True
    allowed_license_ids: frozenset[str] | None = None
    license_policy: ExportLicensePolicy | None = None

    def permits_record(self, record: MiningRecord) -> bool:
        return self.license_policy is None or self.license_policy.allows(record)

    def permits_audio(self, record: MiningRecord) -> bool:
        if not self.include_audio_reference:
            return False
        if self.require_redistribution and record.redistribution is not True:
            return False
        return self.allowed_license_ids is None or record.license_id in self.allowed_license_ids


@dataclass(frozen=True)
class TextResolverExample:
    schema_version: str
    record_id: str
    context: str
    target: str
    candidates: tuple[str, ...]
    gold: str
    confidence: float
    source_type: str
    audio_uri: str | None
    license_id: str | None
    provenance: dict[str, str]

    def as_json(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "record_id": self.record_id,
            "context": self.context,
            "target": self.target,
            "candidates": list(self.candidates),
            "gold": self.gold,
            "confidence": self.confidence,
            "source_type": self.source_type,
            "audio_uri": self.audio_uri,
            "license_id": self.license_id,
            "provenance": self.provenance,
        }


@dataclass(frozen=True)
class TextResolverExportResult:
    jsonl_path: Path
    parquet_path: Path
    dataset_fingerprint: str
    record_count: int


def export_text_resolver(
    records: list[MiningRecord],
    output_dir: str | Path,
    *,
    policy: TextResolverExportPolicy | None = None,
    source_type: str = "audio_mined",
) -> TextResolverExportResult:
    """Export accepted records, omitting non-redistributable media references by policy."""

    active_policy = policy or TextResolverExportPolicy()
    examples = [
        _to_example(record, active_policy, source_type=source_type)
        for record in sorted(records, key=lambda item: item.record_id)
        if record.accepted and active_policy.permits_record(record)
    ]
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    fingerprint = content_fingerprint(
        {
            "record_ids": [item.record_id for item in examples],
            "policy": {
                "include_audio_reference": active_policy.include_audio_reference,
                "require_redistribution": active_policy.require_redistribution,
                "allowed_license_ids": sorted(active_policy.allowed_license_ids)
                if active_policy.allowed_license_ids is not None
                else None,
                "license_policy": active_policy.license_policy.signature()
                if active_policy.license_policy is not None
                else None,
            },
        },
        namespace="text-resolver-dataset",
    )
    jsonl_path = root / "examples.jsonl"
    parquet_path = root / "examples.parquet"
    _write_jsonl(jsonl_path, examples)
    _write_parquet(parquet_path, examples, fingerprint)
    return TextResolverExportResult(jsonl_path, parquet_path, fingerprint, len(examples))


def _to_example(
    record: MiningRecord,
    policy: TextResolverExportPolicy,
    *,
    source_type: str,
) -> TextResolverExample:
    if record.predicted_candidate is None:
        raise ValueError(f"accepted record {record.record_id} has no selected candidate")
    if not 0 <= record.predicted_candidate < len(record.candidates):
        raise ValueError("selected candidate is outside candidate list")
    start, end = record.target_char_start, record.target_char_end
    if start is None or end is None:
        start = record.sentence_original.casefold().find(record.target_word.casefold())
        end = start + len(record.target_word) if start >= 0 else -1
    if start < 0 or end <= start or end > len(record.sentence_original):
        raise ValueError(f"cannot deterministically locate target for {record.record_id}")
    context = (
        record.sentence_original[:start]
        + "<w>"
        + record.sentence_original[start:end]
        + "</w>"
        + record.sentence_original[end:]
    )
    return TextResolverExample(
        schema_version=SCHEMA_VERSION,
        record_id=record.record_id,
        context=context,
        target=record.target_word,
        candidates=tuple(item.stressed_form for item in record.candidates),
        gold=record.candidates[record.predicted_candidate].stressed_form,
        confidence=float(record.calibrated_confidence or 0.0),
        source_type=source_type,
        audio_uri=record.audio_uri if policy.permits_audio(record) else None,
        license_id=record.license_id,
        provenance={**record.provenance, "internal_audio_uri": record.audio_uri},
    )


def _write_jsonl(path: Path, examples: list[TextResolverExample]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            "".join(
                json.dumps(item.as_json(), ensure_ascii=False, sort_keys=True) + "\n"
                for item in examples
            ),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_parquet(path: Path, examples: list[TextResolverExample], fingerprint: str) -> None:
    table = pa.table(
        {
            "schema_version": [item.schema_version for item in examples],
            "record_id": [item.record_id for item in examples],
            "context": [item.context for item in examples],
            "target": [item.target for item in examples],
            "candidates": [list(item.candidates) for item in examples],
            "gold": [item.gold for item in examples],
            "confidence": [item.confidence for item in examples],
            "source_type": [item.source_type for item in examples],
            "audio_uri": [item.audio_uri for item in examples],
            "license_id": [item.license_id for item in examples],
            "provenance_json": [json.dumps(item.provenance, sort_keys=True) for item in examples],
        }
    )
    metadata = {
        **(table.schema.metadata or {}),
        b"ukstress.schema_version": SCHEMA_VERSION.encode(),
        b"ukstress.record_type": b"text_resolver",
        b"ukstress.dataset_fingerprint": fingerprint.encode(),
    }
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        pq.write_table(table.replace_schema_metadata(metadata), temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
