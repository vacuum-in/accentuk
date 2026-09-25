import pyarrow as pa

from ukstress.datasets import CorpusManifestRecord, RejectionRecord
from ukstress.datasets.arrow import (
    RECORD_TYPE_KEY,
    SCHEMA_VERSION_KEY,
    schema_for,
    table_from_records,
)


def test_schema_has_explicit_version_and_record_type() -> None:
    schema = schema_for(CorpusManifestRecord)

    assert schema.metadata[SCHEMA_VERSION_KEY] == b"1.0"
    assert schema.metadata[RECORD_TYPE_KEY] == b"corpus_manifest"
    assert schema.field("schema_version").nullable is False


def test_canonical_record_converts_to_explicit_arrow_table() -> None:
    record = CorpusManifestRecord(
        source_id="fixture",
        utterance_id="utt-1",
        audio_uri="audio.wav",
        transcript="Це замок.",
        provenance={"manifest": "fixture.jsonl"},
    )

    table = table_from_records([record])

    assert isinstance(table, pa.Table)
    assert table.schema == schema_for(CorpusManifestRecord)
    assert table.column("source_id").to_pylist() == ["fixture"]


def test_table_rejects_mixed_record_types() -> None:
    manifest = CorpusManifestRecord(source_id="s", utterance_id="u", audio_uri="a.wav")
    rejection = RejectionRecord(source_id="s", utterance_id="u", stage="decode", reasons=["bad"])

    try:
        table_from_records([manifest, rejection])
    except ValueError as error:
        assert "same canonical type" in str(error)
    else:
        raise AssertionError("mixed canonical types were accepted")
