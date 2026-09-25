"""Decode untrusted local media into a bounded canonical waveform."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import cast
from urllib.parse import unquote, urlparse

import numpy as np
import soundfile as sf

from ukstress.datasets import CorpusManifestRecord
from ukstress.provenance.governance import validate_media_uri


class AudioDecodeError(ValueError):
    def __init__(self, reason: str, detail: str) -> None:
        self.reason = reason
        self.detail = detail
        super().__init__(f"{reason}: {detail}")


@dataclass(frozen=True)
class DecodedAudio:
    waveform: np.ndarray
    sample_rate: int
    duration_s: float
    source_sample_rate: int
    source_channels: int
    source_duration_s: float
    segment_start_s: float
    transformations: tuple[str, ...]


def _resolve_local_path(audio_uri: str, *, base_dir: Path | None) -> Path:
    try:
        validate_media_uri(audio_uri, base_dir=base_dir)
    except ValueError as error:
        raise AudioDecodeError("unsupported_audio_uri", str(error)) from error
    parsed = urlparse(audio_uri)
    if parsed.scheme not in {"", "file"}:
        raise AudioDecodeError(
            "unsupported_audio_uri", "only local paths and file:// URIs are supported"
        )
    if parsed.scheme == "file" and parsed.netloc not in {"", "localhost"}:
        raise AudioDecodeError("unsupported_audio_uri", "remote file URI authorities are forbidden")
    raw_path = unquote(parsed.path) if parsed.scheme == "file" else audio_uri
    path = Path(raw_path)
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    return path


def _resample_linear(waveform: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return waveform
    target_length = max(1, round(len(waveform) * target_rate / source_rate))
    if len(waveform) == 1:
        return np.repeat(waveform, target_length).astype(np.float32, copy=False)
    source_positions = np.arange(len(waveform), dtype=np.float64)
    target_positions = np.linspace(0, len(waveform) - 1, num=target_length, dtype=np.float64)
    interpolated = np.interp(target_positions, source_positions, waveform).astype(np.float32)
    return cast(np.ndarray, interpolated)


def decode_audio(
    record: CorpusManifestRecord,
    *,
    base_dir: str | Path | None = None,
    target_sample_rate: int = 16_000,
    max_duration_s: float = 3600.0,
) -> DecodedAudio:
    """Decode one record without shelling out or modifying the source media."""

    if target_sample_rate <= 0:
        raise ValueError("target_sample_rate must be positive")
    if not math.isfinite(max_duration_s) or max_duration_s <= 0:
        raise ValueError("max_duration_s must be finite and positive")
    path = _resolve_local_path(
        record.audio_uri, base_dir=Path(base_dir) if base_dir is not None else None
    )
    try:
        info = sf.info(path)
    except (OSError, RuntimeError, sf.LibsndfileError) as error:
        raise AudioDecodeError("audio_decode_failed", str(error)) from error
    if info.frames <= 0 or info.samplerate <= 0 or info.channels <= 0:
        raise AudioDecodeError("audio_metadata_invalid", "audio has invalid frame metadata")
    source_duration_s = info.frames / info.samplerate
    start_s = record.segment_start_s or 0.0
    end_s = record.segment_end_s if record.segment_end_s is not None else source_duration_s
    if end_s > source_duration_s + (1 / info.samplerate):
        raise AudioDecodeError("audio_segment_invalid", "segment ends after source duration")
    segment_duration_s = end_s - start_s
    if segment_duration_s <= 0 or segment_duration_s > max_duration_s:
        raise AudioDecodeError(
            "audio_segment_invalid", "segment duration is outside configured bounds"
        )
    start_frame = round(start_s * info.samplerate)
    stop_frame = min(info.frames, round(end_s * info.samplerate))
    try:
        waveform, sample_rate = sf.read(
            path,
            start=start_frame,
            stop=stop_frame,
            dtype="float32",
            always_2d=True,
        )
    except (OSError, RuntimeError, sf.LibsndfileError) as error:
        raise AudioDecodeError("audio_decode_failed", str(error)) from error
    if waveform.shape[0] == 0 or not np.isfinite(waveform).all():
        raise AudioDecodeError("audio_decode_failed", "decoded waveform is empty or non-finite")

    transformations: list[str] = []
    if record.segment_start_s is not None:
        transformations.append("segment_crop")
    mono = waveform.mean(axis=1, dtype=np.float64).astype(np.float32)
    if info.channels != 1:
        transformations.append("downmix_mono")
    if sample_rate != target_sample_rate:
        mono = _resample_linear(mono, sample_rate, target_sample_rate)
        transformations.append(f"resample_{sample_rate}_to_{target_sample_rate}")
    if np.any((mono < -1.0) | (mono > 1.0)):
        mono = np.clip(mono, -1.0, 1.0)
        transformations.append("clip_unit_range")
    mono.setflags(write=False)
    return DecodedAudio(
        waveform=mono,
        sample_rate=target_sample_rate,
        duration_s=len(mono) / target_sample_rate,
        source_sample_rate=info.samplerate,
        source_channels=info.channels,
        source_duration_s=source_duration_s,
        segment_start_s=start_s,
        transformations=tuple(transformations),
    )
