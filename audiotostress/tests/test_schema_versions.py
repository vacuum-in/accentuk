from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pydantic import ValidationError

from ukstress.datasets import CorpusManifestRecord
from ukstress.datasets.arrow import (
    RECORD_TYPE_KEY,
    SCHEMA_VERSION_KEY,
    UnsupportedSchemaVersion,
    require_supported_schema_version,
    schema_for,
)
from ukstress.datasets.parquet import inspect_shard


@pytest.mark.parametrize("version", ["0.9", "2.0", "latest"])
def test_canonical_model_rejects_unknown_schema_version(version: str) -> None:
    with pytest.raises(ValidationError, match=r"Input should be '1\.0'"):
        CorpusManifestRecord(
            schema_version=version,  # type: ignore[arg-type]
            source_id="source",
            utterance_id="utt",
            audio_uri="audio.wav",
        )


def test_arrow_schema_without_version_is_rejected() -> None:
    schema = pa.schema([pa.field("schema_version", pa.string())])

    with pytest.raises(UnsupportedSchemaVersion, match="has no"):
        require_supported_schema_version(schema)


def test_parquet_shard_with_unregistered_old_version_is_rejected(tmp_path: Path) -> None:
    schema = schema_for(CorpusManifestRecord)
    metadata = {
        **schema.metadata,
        SCHEMA_VERSION_KEY: b"0.9",
        RECORD_TYPE_KEY: b"corpus_manifest",
        b"ukstress.input_fingerprint": b"sha256:input",
        b"ukstress.record_count": b"0",
    }
    table = pa.Table.from_pylist([], schema=schema.with_metadata(metadata))
    path = tmp_path / "part-old.parquet"
    pq.write_table(table, path)

    with pytest.raises(UnsupportedSchemaVersion, match=r"0\.9"):
        inspect_shard(path)
