"""Manifest-to-normalized-utterance ingestion stage."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ukstress.audio import AudioDecodeError, DecodedAudio, decode_audio
from ukstress.datasets import (
    CorpusManifestRecord,
    NormalizedUtterance,
    RejectionRecord,
)
from ukstress.datasets.manifest import load_manifest_rows
from ukstress.provenance.fingerprints import content_fingerprint
from ukstress.text import NORMALIZATION_VERSION, normalize_transcript


@dataclass(frozen=True)
class IngestionResult:
    utterance: NormalizedUtterance | None = None
    audio: DecodedAudio | None = None
    rejection: RejectionRecord | None = None

    def __post_init__(self) -> None:
        accepted = self.utterance is not None and self.audio is not None and self.rejection is None
        rejected = self.utterance is None and self.audio is None and self.rejection is not None
        if not (accepted or rejected):
            raise ValueError("ingestion result must be entirely accepted or rejected")


def ingest_record(
    record: CorpusManifestRecord,
    *,
    base_dir: str | Path | None = None,
    target_sample_rate: int = 16_000,
) -> IngestionResult:
    try:
        audio = decode_audio(
            record,
            base_dir=base_dir,
            target_sample_rate=target_sample_rate,
        )
    except AudioDecodeError as error:
        return IngestionResult(
            rejection=RejectionRecord(
                source_id=record.source_id,
                utterance_id=record.utterance_id,
                stage="ingest",
                reasons=[error.reason],
                detail=error.detail,
                provenance=record.provenance,
            )
        )

    original = record.transcript or ""
    normalized = normalize_transcript(original)
    record_fingerprint = content_fingerprint(
        {
            "source_id": record.source_id,
            "utterance_id": record.utterance_id,
            "audio_uri": record.audio_uri,
            "segment_start_s": record.segment_start_s,
            "segment_end_s": record.segment_end_s,
            "sample_rate": target_sample_rate,
            "normalization_version": NORMALIZATION_VERSION,
        },
        namespace="normalized-utterance",
    )
    utterance = NormalizedUtterance(
        record_id=f"utt_{record_fingerprint.removeprefix('sha256:')[:24]}",
        source_id=record.source_id,
        utterance_id=record.utterance_id,
        audio_uri=record.audio_uri,
        transcript_original=original,
        transcript_normalized=normalized,
        speaker_id=record.speaker_id,
        sample_rate=audio.sample_rate,
        duration_s=audio.duration_s,
        source_sample_rate=audio.source_sample_rate,
        source_channels=audio.source_channels,
        segment_start_s=audio.segment_start_s,
        transformations=list(audio.transformations),
        license_id=record.license_id,
        commercial_use=record.commercial_use,
        redistribution=record.redistribution,
        provenance={**record.provenance, "normalization_version": NORMALIZATION_VERSION},
    )
    return IngestionResult(utterance=utterance, audio=audio)


def ingest_manifest(path: str | Path, *, target_sample_rate: int = 16_000) -> list[IngestionResult]:
    manifest_path = Path(path)
    results: list[IngestionResult] = []
    for row in load_manifest_rows(manifest_path):
        if row.rejection is not None:
            results.append(IngestionResult(rejection=row.rejection))
        elif row.record is not None:
            results.append(
                ingest_record(
                    row.record,
                    base_dir=manifest_path.parent,
                    target_sample_rate=target_sample_rate,
                )
            )
    return results
