"""Stable record identifiers and dataset lineage fingerprints."""

from __future__ import annotations

import math
import unicodedata
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

from ukstress.datasets.models import SCHEMA_VERSION
from ukstress.provenance.fingerprints import content_fingerprint


def quantize_milliseconds(seconds: float) -> int:
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("timestamp must be finite and non-negative")
    milliseconds = Decimal(str(seconds)) * 1000
    return int(milliseconds.quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def full_record_fingerprint(
    *,
    source_id: str,
    utterance_id: str,
    normalized_target: str,
    word_start_s: float,
    word_end_s: float,
    schema_version: str = SCHEMA_VERSION,
) -> str:
    if not source_id or not utterance_id or not normalized_target:
        raise ValueError("record identity fields must be non-empty")
    start_ms = quantize_milliseconds(word_start_s)
    end_ms = quantize_milliseconds(word_end_s)
    if start_ms >= end_ms:
        raise ValueError("quantized word start must be before end")
    identity = {
        "schema_version": schema_version,
        "source_id": source_id,
        "utterance_id": utterance_id,
        "target": unicodedata.normalize("NFC", normalized_target).casefold(),
        "word_start_ms": start_ms,
        "word_end_ms": end_ms,
    }
    return content_fingerprint(identity, namespace="mining-record-id")


def stable_record_id(**identity: Any) -> str:
    """Return a readable 96-bit prefix while the full hash remains reproducible."""

    fingerprint = full_record_fingerprint(**identity)
    return f"rec_{fingerprint.removeprefix('sha256:')[:24]}"


def dataset_fingerprint(
    *,
    ordered_record_ids: list[str],
    lexicon_fingerprint: str,
    normalization_version: str,
    split_config: dict[str, Any],
    schema_version: str = SCHEMA_VERSION,
) -> str:
    """Bind ordered inputs and all label/split semantics into a dataset identity."""

    return content_fingerprint(
        {
            "schema_version": schema_version,
            "ordered_record_ids": ordered_record_ids,
            "lexicon_fingerprint": lexicon_fingerprint,
            "normalization_version": normalization_version,
            "split_config": split_config,
        },
        namespace="dataset",
    )
