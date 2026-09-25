"""Deterministic compressed staging artifacts and parser checkpoints."""

from __future__ import annotations

import io
import json
import os
import platform
import sys
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import zstandard


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode()


def write_jsonl_zst(path: Path, records: list[dict[str, Any]]) -> int:
    """Write records sorted by stable natural key with deterministic compression."""
    ordered = sorted(records, key=lambda record: str(record["natural_key"]))
    keys = [str(record["natural_key"]) for record in ordered]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate natural key in staging records")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    compressor = zstandard.ZstdCompressor(level=9, threads=0, write_checksum=True)
    with temporary.open("wb") as raw:
        with compressor.stream_writer(raw, closefd=False) as output:
            for record in ordered:
                output.write(_canonical_json(record))
        raw.flush()
        os.fsync(raw.fileno())
    os.replace(temporary, path)
    return len(ordered)


def write_jsonl_zst_stream(path: Path, records: Iterable[dict[str, Any]]) -> int:
    """Write already natural-key-ordered records without materializing them."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    compressor = zstandard.ZstdCompressor(level=9, threads=0, write_checksum=True)
    count = 0
    previous_key: str | None = None
    with temporary.open("wb") as raw:
        with compressor.stream_writer(raw, closefd=False) as output:
            for record in records:
                key = str(record["natural_key"])
                if previous_key is not None and key <= previous_key:
                    raise ValueError("streaming staging records are not uniquely ordered")
                output.write(_canonical_json(record))
                previous_key = key
                count += 1
        raw.flush()
        os.fsync(raw.fileno())
    os.replace(temporary, path)
    return count


def iter_jsonl_zst(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("rb") as raw:
        with zstandard.ZstdDecompressor().stream_reader(raw) as source:
            with io.TextIOWrapper(source, encoding="utf-8") as text:
                for line in text:
                    yield json.loads(line)


def read_jsonl_zst(path: Path) -> list[dict[str, Any]]:
    return list(iter_jsonl_zst(path))


@dataclass(frozen=True)
class ParserCheckpoint:
    last_page_id: int
    records_written: int
    complete: bool = False


def write_checkpoint(path: Path, checkpoint: ParserCheckpoint) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(_canonical_json(asdict(checkpoint)))
    os.replace(temporary, path)


def read_checkpoint(path: Path) -> ParserCheckpoint | None:
    if not path.exists():
        return None
    return ParserCheckpoint(**json.loads(path.read_text(encoding="utf-8")))


@dataclass(frozen=True)
class ImportManifest:
    dataset_key: str
    dump_url: str
    dump_sha256: str
    dump_timestamp: str | None
    parser_version: str
    normalization_version: str
    schema_version: str
    git_commit: str | None
    command_options: dict[str, Any]
    started_at: str
    finished_at: str
    statistics: dict[str, int]
    python_version: str = platform.python_version()
    platform: str = platform.platform()


def write_manifest(path: Path, manifest: ImportManifest) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(manifest)
    payload["implementation"] = sys.implementation.name
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(_canonical_json(payload))
    os.replace(temporary, path)
