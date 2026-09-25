"""Atomic, idempotent Parquet shard storage."""

from __future__ import annotations

import os
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq

from ukstress.datasets.arrow import (
    RECORD_TYPE_KEY,
    SCHEMA_VERSION_KEY,
    require_supported_schema_version,
    table_from_records,
)
from ukstress.datasets.models import CanonicalModel

INPUT_FINGERPRINT_KEY = b"ukstress.input_fingerprint"
RECORD_COUNT_KEY = b"ukstress.record_count"
_SHARD_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class ShardConflictError(RuntimeError):
    """An existing shard belongs to different logical input."""


@dataclass(frozen=True)
class ShardStatus:
    path: Path
    shard_id: str
    schema_version: str
    record_type: str
    input_fingerprint: str
    record_count: int


@dataclass(frozen=True)
class ShardWriteResult:
    status: ShardStatus
    skipped: bool


def shard_path(output_dir: str | Path, shard_id: str) -> Path:
    if not _SHARD_ID.fullmatch(shard_id):
        raise ValueError("shard_id must contain only letters, digits, dot, underscore, and hyphen")
    return Path(output_dir) / f"part-{shard_id}.parquet"


def inspect_shard(path: str | Path) -> ShardStatus:
    shard = Path(path)
    match = re.fullmatch(r"part-(.+)\.parquet", shard.name)
    if match is None:
        raise ValueError(f"not a canonical shard path: {shard}")
    schema = pq.read_schema(shard)
    require_supported_schema_version(schema)
    metadata = schema.metadata or {}
    required = (SCHEMA_VERSION_KEY, RECORD_TYPE_KEY, INPUT_FINGERPRINT_KEY, RECORD_COUNT_KEY)
    missing = [key.decode("ascii") for key in required if key not in metadata]
    if missing:
        raise ValueError(f"shard metadata is incomplete: {', '.join(missing)}")
    return ShardStatus(
        path=shard,
        shard_id=match.group(1),
        schema_version=metadata[SCHEMA_VERSION_KEY].decode("ascii"),
        record_type=metadata[RECORD_TYPE_KEY].decode("ascii"),
        input_fingerprint=metadata[INPUT_FINGERPRINT_KEY].decode("ascii"),
        record_count=int(metadata[RECORD_COUNT_KEY].decode("ascii")),
    )


def completed_shards(output_dir: str | Path) -> dict[str, ShardStatus]:
    result: dict[str, ShardStatus] = {}
    for path in sorted(Path(output_dir).glob("part-*.parquet")):
        status = inspect_shard(path)
        result[status.shard_id] = status
    return result


def is_shard_complete(output_dir: str | Path, *, shard_id: str, input_fingerprint: str) -> bool:
    path = shard_path(output_dir, shard_id)
    if not path.exists():
        return False
    return inspect_shard(path).input_fingerprint == input_fingerprint


def write_parquet_shard(
    records: Sequence[CanonicalModel],
    output_dir: str | Path,
    *,
    shard_id: str,
    input_fingerprint: str,
    force: bool = False,
) -> ShardWriteResult:
    """Atomically commit a shard, or skip an already matching shard."""

    if not records:
        raise ValueError("a Parquet shard must contain at least one record")
    if not input_fingerprint:
        raise ValueError("input_fingerprint must be non-empty")
    destination = shard_path(output_dir, shard_id)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and not force:
        status = inspect_shard(destination)
        if status.input_fingerprint == input_fingerprint:
            return ShardWriteResult(status=status, skipped=True)
        raise ShardConflictError(
            f"shard {shard_id} already exists for input {status.input_fingerprint}"
        )

    table = table_from_records(list(records))
    metadata = {
        **(table.schema.metadata or {}),
        INPUT_FINGERPRINT_KEY: input_fingerprint.encode("ascii"),
        RECORD_COUNT_KEY: str(table.num_rows).encode("ascii"),
    }
    table = table.replace_schema_metadata(metadata)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        pq.write_table(table, temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    return ShardWriteResult(status=inspect_shard(destination), skipped=False)
