"""Explicit Arrow schemas for versioned canonical datasets."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, cast

import pyarrow as pa

from ukstress.datasets.models import (
    SCHEMA_VERSION,
    CanonicalModel,
    CorpusManifestRecord,
    FeatureExample,
    MiningRecord,
    NormalizedUtterance,
    RejectionRecord,
    WordAlignment,
)

SCHEMA_VERSION_KEY = b"ukstress.schema_version"
RECORD_TYPE_KEY = b"ukstress.record_type"


class UnsupportedSchemaVersion(ValueError):
    """A persisted dataset cannot be safely read by this package version."""


def require_supported_schema_version(schema: pa.Schema) -> str:
    metadata = schema.metadata or {}
    raw_version = metadata.get(SCHEMA_VERSION_KEY)
    if raw_version is None:
        raise UnsupportedSchemaVersion("Arrow schema has no ukstress.schema_version metadata")
    version = cast(bytes, raw_version).decode("ascii")
    if version != SCHEMA_VERSION:
        raise UnsupportedSchemaVersion(
            f"unsupported schema version {version!r}; this build supports {SCHEMA_VERSION!r}"
        )
    return version


def _metadata(record_type: str) -> dict[bytes, bytes]:
    return {
        SCHEMA_VERSION_KEY: SCHEMA_VERSION.encode("ascii"),
        RECORD_TYPE_KEY: record_type.encode("ascii"),
    }


STRING_MAP = pa.map_(pa.string(), pa.string())
VOWEL_STRUCT = pa.struct(
    [
        pa.field("schema_version", pa.string(), nullable=False),
        pa.field("vowel_index", pa.int32(), nullable=False),
        pa.field("grapheme", pa.string(), nullable=False),
        pa.field("phone", pa.string()),
        pa.field("start_s", pa.float64(), nullable=False),
        pa.field("end_s", pa.float64(), nullable=False),
        pa.field("confidence", pa.float64()),
    ]
)
STRESS_CANDIDATE_STRUCT = pa.struct(
    [
        pa.field("stressed_form", pa.string(), nullable=False),
        pa.field("vowel_index", pa.int32(), nullable=False),
        pa.field("source", pa.string(), nullable=False),
    ]
)

CORPUS_MANIFEST_SCHEMA = pa.schema(
    [
        pa.field("schema_version", pa.string(), nullable=False),
        pa.field("source_id", pa.string(), nullable=False),
        pa.field("utterance_id", pa.string(), nullable=False),
        pa.field("audio_uri", pa.string(), nullable=False),
        pa.field("transcript", pa.string()),
        pa.field("speaker_id", pa.string()),
        pa.field("segment_start_s", pa.float64()),
        pa.field("segment_end_s", pa.float64()),
        pa.field("language", pa.string(), nullable=False),
        pa.field("license_id", pa.string()),
        pa.field("commercial_use", pa.bool_()),
        pa.field("redistribution", pa.bool_()),
        pa.field("provenance", STRING_MAP, nullable=False),
    ],
    metadata=_metadata("corpus_manifest"),
)

NORMALIZED_UTTERANCE_SCHEMA = pa.schema(
    [
        pa.field("schema_version", pa.string(), nullable=False),
        pa.field("record_id", pa.string(), nullable=False),
        pa.field("source_id", pa.string(), nullable=False),
        pa.field("utterance_id", pa.string(), nullable=False),
        pa.field("audio_uri", pa.string(), nullable=False),
        pa.field("transcript_original", pa.string(), nullable=False),
        pa.field("transcript_normalized", pa.string(), nullable=False),
        pa.field("speaker_id", pa.string()),
        pa.field("sample_rate", pa.int32(), nullable=False),
        pa.field("duration_s", pa.float64(), nullable=False),
        pa.field("source_sample_rate", pa.int32(), nullable=False),
        pa.field("source_channels", pa.int32(), nullable=False),
        pa.field("segment_start_s", pa.float64(), nullable=False),
        pa.field("transformations", pa.list_(pa.string()), nullable=False),
        pa.field("license_id", pa.string()),
        pa.field("commercial_use", pa.bool_()),
        pa.field("redistribution", pa.bool_()),
        pa.field("provenance", STRING_MAP, nullable=False),
    ],
    metadata=_metadata("normalized_utterance"),
)

WORD_ALIGNMENT_SCHEMA = pa.schema(
    [
        pa.field("schema_version", pa.string(), nullable=False),
        pa.field("token", pa.string(), nullable=False),
        pa.field("normalized_token", pa.string(), nullable=False),
        pa.field("start_s", pa.float64(), nullable=False),
        pa.field("end_s", pa.float64(), nullable=False),
        pa.field("confidence", pa.float64()),
        pa.field("char_start", pa.int32()),
        pa.field("char_end", pa.int32()),
        pa.field("backend", pa.string(), nullable=False),
        pa.field("model_id", pa.string()),
    ],
    metadata=_metadata("word_alignment"),
)

FEATURE_EXAMPLE_SCHEMA = pa.schema(
    [
        pa.field("schema_version", pa.string(), nullable=False),
        pa.field("record_id", pa.string(), nullable=False),
        pa.field("source_id", pa.string(), nullable=False),
        pa.field("utterance_id", pa.string(), nullable=False),
        pa.field("speaker_id", pa.string()),
        pa.field("audio_uri", pa.string(), nullable=False),
        pa.field("target_word", pa.string(), nullable=False),
        pa.field("word_start_s", pa.float64(), nullable=False),
        pa.field("word_end_s", pa.float64(), nullable=False),
        pa.field("vowels", pa.list_(VOWEL_STRUCT), nullable=False),
        pa.field("gold_vowel_index", pa.int32(), nullable=False),
        pa.field("ssl_features", pa.list_(pa.list_(pa.float32()))),
        pa.field("prosodic_features_json", pa.list_(pa.string()), nullable=False),
        pa.field("lexicon_fingerprint", pa.string(), nullable=False),
        pa.field("config_fingerprint", pa.string(), nullable=False),
        pa.field("license_id", pa.string()),
        pa.field("provenance", STRING_MAP, nullable=False),
    ],
    metadata=_metadata("feature_example"),
)

REJECTION_SCHEMA = pa.schema(
    [
        pa.field("schema_version", pa.string(), nullable=False),
        pa.field("source_id", pa.string(), nullable=False),
        pa.field("utterance_id", pa.string(), nullable=False),
        pa.field("record_id", pa.string()),
        pa.field("stage", pa.string(), nullable=False),
        pa.field("reasons", pa.list_(pa.string()), nullable=False),
        pa.field("detail", pa.string()),
        pa.field("provenance", STRING_MAP, nullable=False),
    ],
    metadata=_metadata("rejection"),
)

MINING_RECORD_SCHEMA = pa.schema(
    [
        pa.field("schema_version", pa.string(), nullable=False),
        pa.field("record_id", pa.string(), nullable=False),
        pa.field("source_id", pa.string(), nullable=False),
        pa.field("utterance_id", pa.string(), nullable=False),
        pa.field("speaker_id", pa.string()),
        pa.field("audio_uri", pa.string(), nullable=False),
        pa.field("sentence_original", pa.string(), nullable=False),
        pa.field("sentence_normalized", pa.string(), nullable=False),
        pa.field("target_word", pa.string(), nullable=False),
        pa.field("target_char_start", pa.int32()),
        pa.field("target_char_end", pa.int32()),
        pa.field("word_start_s", pa.float64(), nullable=False),
        pa.field("word_end_s", pa.float64(), nullable=False),
        pa.field("vowels", pa.list_(VOWEL_STRUCT), nullable=False),
        pa.field("candidates", pa.list_(STRESS_CANDIDATE_STRUCT), nullable=False),
        pa.field("ranker_probs", pa.list_(pa.float32())),
        pa.field("candidate_scorer_probs", pa.list_(pa.float32())),
        pa.field("stress_ctc_candidate", pa.int32()),
        pa.field("predicted_candidate", pa.int32()),
        pa.field("calibrated_confidence", pa.float64()),
        pa.field("candidate_margin", pa.float64()),
        pa.field("alignment_score", pa.float64()),
        pa.field("identity_score", pa.float64()),
        pa.field("accepted", pa.bool_(), nullable=False),
        pa.field("rejection_reasons", pa.list_(pa.string()), nullable=False),
        pa.field("lexicon_fingerprint", pa.string(), nullable=False),
        pa.field("model_versions", STRING_MAP, nullable=False),
        pa.field("config_fingerprint", pa.string(), nullable=False),
        pa.field("license_id", pa.string()),
        pa.field("commercial_use", pa.bool_()),
        pa.field("redistribution", pa.bool_()),
        pa.field("provenance", STRING_MAP, nullable=False),
    ],
    metadata=_metadata("mining_record"),
)

_SCHEMAS: dict[type[CanonicalModel], pa.Schema] = {
    CorpusManifestRecord: CORPUS_MANIFEST_SCHEMA,
    NormalizedUtterance: NORMALIZED_UTTERANCE_SCHEMA,
    WordAlignment: WORD_ALIGNMENT_SCHEMA,
    FeatureExample: FEATURE_EXAMPLE_SCHEMA,
    MiningRecord: MINING_RECORD_SCHEMA,
    RejectionRecord: REJECTION_SCHEMA,
}


def schema_for(record_type: type[CanonicalModel]) -> pa.Schema:
    try:
        return _SCHEMAS[record_type]
    except KeyError as error:
        raise ValueError(f"no Arrow schema registered for {record_type.__name__}") from error


def storage_dict(record: CanonicalModel) -> dict[str, Any]:
    """Convert a canonical model into its explicit Arrow storage representation."""

    payload = record.model_dump(mode="python")
    if isinstance(record, FeatureExample):
        features = payload.pop("prosodic_features")
        payload["prosodic_features_json"] = [
            json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            for item in features
        ]
    return payload


def table_from_records(records: list[CanonicalModel]) -> pa.Table:
    if not records:
        raise ValueError("cannot infer a schema from an empty record list")
    record_type = type(records[0])
    if any(type(record) is not record_type for record in records):
        raise ValueError("all records in a table must have the same canonical type")
    schema = schema_for(record_type)
    return pa.Table.from_pylist([storage_dict(record) for record in records], schema=schema)


def schema_metadata(schema: pa.Schema) -> Mapping[bytes, bytes]:
    return schema.metadata or {}
