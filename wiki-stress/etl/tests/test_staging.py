from pathlib import Path

import pytest

from ukstress.staging import (
    ImportManifest,
    ParserCheckpoint,
    read_checkpoint,
    read_jsonl_zst,
    write_checkpoint,
    write_jsonl_zst,
    write_manifest,
)


def test_jsonl_zst_is_deterministic_and_replayable(tmp_path: Path) -> None:
    records = [
        {"natural_key": "b", "value": "мо́ва"},
        {"natural_key": "a", "value": "за́мок"},
    ]
    first = tmp_path / "first.jsonl.zst"
    second = tmp_path / "second.jsonl.zst"

    assert write_jsonl_zst(first, records) == 2
    write_jsonl_zst(second, list(reversed(records)))

    assert first.read_bytes() == second.read_bytes()
    assert [row["natural_key"] for row in read_jsonl_zst(first)] == ["a", "b"]


def test_duplicate_natural_key_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="duplicate natural key"):
        write_jsonl_zst(
            tmp_path / "bad.jsonl.zst",
            [{"natural_key": "a"}, {"natural_key": "a"}],
        )


def test_checkpoint_round_trip_and_manifest_versions(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "checkpoint.json"
    checkpoint = ParserCheckpoint(42, 100)
    write_checkpoint(checkpoint_path, checkpoint)
    assert read_checkpoint(checkpoint_path) == checkpoint

    manifest_path = tmp_path / "manifest.json"
    write_manifest(
        manifest_path,
        ImportManifest(
            dataset_key="fixture",
            dump_url="https://example.invalid/dump.bz2",
            dump_sha256="0" * 64,
            dump_timestamp=None,
            parser_version="1",
            normalization_version="1",
            schema_version="1",
            git_commit=None,
            command_options={"workers": 1},
            started_at="2026-01-01T00:00:00Z",
            finished_at="2026-01-01T00:01:00Z",
            statistics={"pages": 10},
        ),
    )
    contents = manifest_path.read_text(encoding="utf-8")
    assert '"dump_sha256":"' + "0" * 64 + '"' in contents
    assert '"python_version":' in contents

