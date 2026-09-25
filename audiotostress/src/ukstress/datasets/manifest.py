"""Versioned corpus manifest loading with row-level validation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from ukstress.datasets.models import CorpusManifestRecord, RejectionRecord


@dataclass(frozen=True)
class ManifestRow:
    line_number: int
    record: CorpusManifestRecord | None = None
    rejection: RejectionRecord | None = None

    def __post_init__(self) -> None:
        if (self.record is None) == (self.rejection is None):
            raise ValueError("manifest row must contain exactly one record or rejection")


class ManifestValidationError(ValueError):
    def __init__(self, rejections: list[RejectionRecord]) -> None:
        self.rejections = rejections
        super().__init__(f"corpus manifest contains {len(rejections)} invalid row(s)")


def load_manifest_rows(path: str | Path) -> list[ManifestRow]:
    manifest_path = Path(path)
    raw_rows = _read_rows(manifest_path)
    rows: list[ManifestRow] = []
    seen: set[tuple[str, str]] = set()
    for line_number, raw in enumerate(raw_rows, start=1):
        source_id = str(raw.get("source_id") or f"<unknown-source:{line_number}>")
        utterance_id = str(raw.get("utterance_id") or f"<unknown-utterance:{line_number}>")
        try:
            record = CorpusManifestRecord.model_validate(raw)
        except ValidationError as error:
            rows.append(
                ManifestRow(
                    line_number=line_number,
                    rejection=RejectionRecord(
                        source_id=source_id,
                        utterance_id=utterance_id,
                        stage="manifest",
                        reasons=["metadata_invalid"],
                        detail=str(error),
                        provenance={"manifest_path": str(manifest_path), "row": str(line_number)},
                    ),
                )
            )
            continue
        key = (record.source_id, record.utterance_id)
        if key in seen:
            rows.append(
                ManifestRow(
                    line_number=line_number,
                    rejection=RejectionRecord(
                        source_id=record.source_id,
                        utterance_id=record.utterance_id,
                        stage="manifest",
                        reasons=["duplicate_manifest_record"],
                        provenance={"manifest_path": str(manifest_path), "row": str(line_number)},
                    ),
                )
            )
            continue
        seen.add(key)
        rows.append(ManifestRow(line_number=line_number, record=record))
    return rows


def load_manifest(path: str | Path) -> list[CorpusManifestRecord]:
    """Load a fully valid manifest, raising with all row rejections otherwise."""

    rows = load_manifest_rows(path)
    rejections = [row.rejection for row in rows if row.rejection is not None]
    if rejections:
        raise ManifestValidationError(rejections)
    return [row.record for row in rows if row.record is not None]


def _read_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.casefold()
    if suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                message = f"manifest line {line_number} is invalid JSON: {error.msg}"
                raise ValueError(message) from error
            if not isinstance(row, dict):
                raise ValueError(f"manifest line {line_number} must be a JSON object")
            rows.append(row)
        return rows
    if suffix in {".json", ".yaml", ".yml"}:
        text = path.read_text(encoding="utf-8")
        document = json.loads(text) if suffix == ".json" else yaml.safe_load(text)
        if isinstance(document, dict):
            document = document.get("records")
        if not isinstance(document, list) or not all(isinstance(row, dict) for row in document):
            raise ValueError("manifest must be a list or a mapping containing a records list")
        return document
    raise ValueError(f"unsupported manifest format {suffix!r}")
