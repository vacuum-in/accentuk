from pathlib import Path

import pyarrow.parquet as pq
import pytest

from ukstress.datasets import CorpusManifestRecord
from ukstress.datasets.parquet import (
    ShardConflictError,
    completed_shards,
    is_shard_complete,
    write_parquet_shard,
)


def _records() -> list[CorpusManifestRecord]:
    return [
        CorpusManifestRecord(source_id="fixture", utterance_id="u1", audio_uri="one.wav"),
        CorpusManifestRecord(source_id="fixture", utterance_id="u2", audio_uri="two.wav"),
    ]


def test_write_parquet_shard_is_atomic_and_resumable(tmp_path: Path) -> None:
    first = write_parquet_shard(
        _records(), tmp_path, shard_id="0001", input_fingerprint="sha256:input"
    )
    second = write_parquet_shard(
        _records(), tmp_path, shard_id="0001", input_fingerprint="sha256:input"
    )

    assert first.skipped is False
    assert second.skipped is True
    assert first.status.record_count == 2
    assert is_shard_complete(
        tmp_path, shard_id="0001", input_fingerprint="sha256:input"
    )
    assert pq.read_table(first.status.path).num_rows == 2
    assert list(tmp_path.glob("*.tmp")) == []


def test_write_parquet_shard_rejects_conflicting_resume(tmp_path: Path) -> None:
    write_parquet_shard(_records(), tmp_path, shard_id="a", input_fingerprint="sha256:old")

    with pytest.raises(ShardConflictError, match="already exists"):
        write_parquet_shard(
            _records(), tmp_path, shard_id="a", input_fingerprint="sha256:new"
        )


def test_completed_shards_reports_committed_files(tmp_path: Path) -> None:
    write_parquet_shard(_records(), tmp_path, shard_id="2", input_fingerprint="sha256:two")
    write_parquet_shard(_records(), tmp_path, shard_id="1", input_fingerprint="sha256:one")

    assert list(completed_shards(tmp_path)) == ["1", "2"]


def test_shard_id_cannot_escape_output_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="shard_id"):
        write_parquet_shard(
            _records(), tmp_path, shard_id="../escape", input_fingerprint="sha256:input"
        )
