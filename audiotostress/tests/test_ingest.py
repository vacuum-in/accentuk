import json
import wave
from array import array
from pathlib import Path

import pytest

from ukstress.audio import AudioDecodeError, decode_audio
from ukstress.datasets import CorpusManifestRecord
from ukstress.pipeline.ingest import ingest_manifest
from ukstress.text import NORMALIZATION_VERSION, normalize_transcript


def _write_stereo_wav(path: Path, *, sample_rate: int = 8_000) -> None:
    frames = array("h")
    for index in range(sample_rate):
        sample = 10_000 if index % 20 < 10 else -10_000
        frames.extend((sample, sample // 2))
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(2)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate)
        stream.writeframes(frames.tobytes())


def test_transcript_normalization_is_deterministic_and_does_not_resolve_stress() -> None:
    text = "  Працівник перевірив ЗА́МОК…  "

    assert normalize_transcript(text) == "працівник перевірив замок"
    assert normalize_transcript("об’єкт — будь-який") == "об'єкт будь-який"
    assert NORMALIZATION_VERSION == "uk-transcript-v1"


def test_decode_audio_downmixes_and_resamples_without_overwriting(tmp_path: Path) -> None:
    audio_path = tmp_path / "stereo.wav"
    _write_stereo_wav(audio_path)
    original = audio_path.read_bytes()
    record = CorpusManifestRecord(
        source_id="fixture", utterance_id="u1", audio_uri=str(audio_path)
    )

    decoded = decode_audio(record)

    assert decoded.waveform.ndim == 1
    assert decoded.sample_rate == 16_000
    assert decoded.duration_s == pytest.approx(1.0, abs=1 / 16_000)
    assert decoded.transformations == ("downmix_mono", "resample_8000_to_16000")
    assert decoded.waveform.flags.writeable is False
    assert audio_path.read_bytes() == original


def test_decode_audio_rejects_non_local_uri() -> None:
    record = CorpusManifestRecord(
        source_id="fixture", utterance_id="u1", audio_uri="https://example.test/audio.wav"
    )

    with pytest.raises(AudioDecodeError) as captured:
        decode_audio(record)
    assert captured.value.reason == "unsupported_audio_uri"


def test_tiny_fixture_ingestion_preserves_text_and_structures_failures(tmp_path: Path) -> None:
    audio_path = tmp_path / "tiny.wav"
    _write_stereo_wav(audio_path)
    manifest_path = tmp_path / "manifest.jsonl"
    rows = [
        {
            "source_id": "fixture",
            "utterance_id": "good",
            "audio_uri": "tiny.wav",
            "transcript": "Працівник перевірив замок.",
            "license_id": "CC0-1.0",
        },
        {
            "source_id": "fixture",
            "utterance_id": "bad-audio",
            "audio_uri": "missing.wav",
            "transcript": "Замок.",
        },
        {"source_id": "fixture", "utterance_id": "bad-metadata", "audio_uri": ""},
    ]
    manifest_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )

    results = ingest_manifest(manifest_path)

    assert results[0].utterance is not None
    assert results[0].utterance.transcript_original == "Працівник перевірив замок."
    assert results[0].utterance.transcript_normalized == "працівник перевірив замок"
    assert results[0].utterance.license_id == "CC0-1.0"
    assert results[1].rejection is not None
    assert results[1].rejection.reasons == ["audio_decode_failed"]
    assert results[2].rejection is not None
    assert results[2].rejection.reasons == ["metadata_invalid"]
