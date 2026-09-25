import json
from pathlib import Path

import pytest

from ukstress.datasets.manifest import (
    ManifestValidationError,
    load_manifest,
    load_manifest_rows,
)


def test_manifest_loads_valid_versioned_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "corpus.jsonl"
    path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "source_id": "fixture",
                "utterance_id": "u1",
                "audio_uri": "one.wav",
                "transcript": "Це замок.",
                "provenance": {"collection": "test"},
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    records = load_manifest(path)

    assert records[0].transcript == "Це замок."
    assert records[0].provenance["collection"] == "test"


def test_manifest_reports_bad_metadata_and_duplicates_as_rejections(tmp_path: Path) -> None:
    path = tmp_path / "corpus.jsonl"
    rows = [
        {"source_id": "fixture", "utterance_id": "u1", "audio_uri": "one.wav"},
        {"source_id": "fixture", "utterance_id": "u1", "audio_uri": "two.wav"},
        {"source_id": "fixture", "utterance_id": "u2", "audio_uri": ""},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    loaded = load_manifest_rows(path)

    assert loaded[0].record is not None
    assert loaded[1].rejection is not None
    assert loaded[1].rejection.reasons == ["duplicate_manifest_record"]
    assert loaded[2].rejection is not None
    assert loaded[2].rejection.reasons == ["metadata_invalid"]
    with pytest.raises(ManifestValidationError) as captured:
        load_manifest(path)
    assert len(captured.value.rejections) == 2
