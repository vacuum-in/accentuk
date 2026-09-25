"""Helpers for carrying source licensing and provenance into derived artifacts."""

from __future__ import annotations

from collections.abc import Mapping

from ukstress.datasets import CorpusManifestRecord


def derived_provenance(
    source: CorpusManifestRecord,
    *,
    stage: str,
    extra: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Build a non-lossy provenance mapping for any derived record."""

    if not stage.strip():
        raise ValueError("stage must be non-empty")
    result = {
        **source.provenance,
        "source_id": source.source_id,
        "utterance_id": source.utterance_id,
        "audio_uri": source.audio_uri,
        "stage": stage,
    }
    if source.license_id is not None:
        result["license_id"] = source.license_id
    if source.commercial_use is not None:
        result["commercial_use"] = str(source.commercial_use).lower()
    if source.redistribution is not None:
        result["redistribution"] = str(source.redistribution).lower()
    result.update(extra or {})
    return result
