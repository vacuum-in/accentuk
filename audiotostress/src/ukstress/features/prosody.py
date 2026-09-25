"""Deterministic duration, energy, and F0 features for vowel intervals."""

from __future__ import annotations

import math

import numpy as np

from ukstress.datasets import VowelInterval


def extract_prosodic_features(
    waveform: np.ndarray,
    sample_rate: int,
    vowels: list[VowelInterval],
    *,
    word_start_s: float,
    word_end_s: float,
    utterance_vowels: list[VowelInterval] | None = None,
    f0_cache: tuple[np.ndarray, np.ndarray] | None = None,
    f0_min_hz: float = 60.0,
    f0_max_hz: float = 400.0,
    f0_frame_ms: int = 40,
    f0_hop_ms: int = 10,
) -> list[dict[str, float | bool | None]]:
    """Extract auditable prosodic features with an explicit F0 validity flag."""

    if waveform.ndim != 1 or sample_rate < 1:
        raise ValueError("waveform must be mono and sample_rate positive")
    if word_start_s < 0.0 or word_start_s >= word_end_s:
        raise ValueError("word bounds must be ordered and non-negative")
    if f0_min_hz <= 0.0 or f0_min_hz >= f0_max_hz:
        raise ValueError("F0 bounds must be positive and ordered")
    durations = [interval.end_s - interval.start_s for interval in vowels]
    utterance_durations = [item.end_s - item.start_s for item in utterance_vowels or []]
    utterance_median = float(np.median(utterance_durations)) if utterance_durations else None
    word_samples = _slice(waveform, sample_rate, word_start_s, word_end_s)
    word_rms = _rms(word_samples)
    # The track covers the whole waveform and does not depend on which word is
    # being measured, but it was recomputed for every one of them: on a
    # two-minute window with 115 words that is 3.8 hours of autocorrelation to
    # describe 60 seconds of speech, and it made prosody the single most
    # expensive stage at 81 s a window regardless of anything else. A caller
    # that measures many words in one window passes the track in once.
    if f0_cache is not None:
        f0_times, f0_values = f0_cache
    else:
        f0_times, f0_values = f0_track(
            waveform,
            sample_rate,
            min_hz=f0_min_hz,
            max_hz=f0_max_hz,
            frame_ms=f0_frame_ms,
            hop_ms=f0_hop_ms,
        )
    features: list[dict[str, float | bool | None]] = []
    for index, interval in enumerate(vowels):
        duration = durations[index]
        others = durations[:index] + durations[index + 1 :]
        samples = _slice(waveform, sample_rate, interval.start_s, interval.end_s)
        rms = _rms(samples)
        voiced = f0_values[(f0_times >= interval.start_s) & (f0_times <= interval.end_s)]
        voiced = voiced[np.isfinite(voiced)]
        f0_valid = len(voiced) > 0
        f0_median = float(np.median(voiced)) if f0_valid else None
        f0_range = float(np.max(voiced) - np.min(voiced)) if f0_valid else None
        f0_slope = _f0_slope(f0_times, f0_values, interval) if f0_valid else None
        features.append(
            {
                "duration_s": duration,
                "log_duration": math.log(duration),
                "duration_relative_word": duration / float(np.mean(others)) if others else None,
                "duration_relative_utterance": (
                    duration / utterance_median if utterance_median else None
                ),
                "rms_energy": rms,
                "peak_energy": float(np.max(np.abs(samples))) if len(samples) else 0.0,
                "relative_rms_energy": rms / word_rms if word_rms else None,
                "f0_median_hz": f0_median,
                "f0_range_hz": f0_range,
                "f0_slope_hz_per_s": f0_slope,
                "f0_valid": f0_valid,
            }
        )
    return features


def f0_track(
    waveform: np.ndarray,
    sample_rate: int,
    *,
    min_hz: float,
    max_hz: float,
    frame_ms: int = 40,
    hop_ms: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate an F0 track with normalized autocorrelation; NaN denotes unvoiced frames."""

    frame_size = round(sample_rate * frame_ms / 1000)
    hop_size = round(sample_rate * hop_ms / 1000)
    if frame_size < 2 or hop_size < 1:
        raise ValueError("F0 frame and hop sizes are too small")
    min_lag = max(1, int(sample_rate / max_hz))
    max_lag = max(min_lag + 1, int(sample_rate / min_hz))
    times: list[float] = []
    values: list[float] = []
    for start in range(0, max(0, len(waveform) - frame_size + 1), hop_size):
        frame = waveform[start : start + frame_size].astype(np.float64, copy=False)
        frame -= np.mean(frame)
        energy = float(np.dot(frame, frame))
        f0 = math.nan
        if energy > 1e-8 and max_lag < len(frame):
            correlation = np.correlate(frame, frame, mode="full")[len(frame) - 1 :]
            normalized = correlation[min_lag : max_lag + 1] / energy
            lag = min_lag + int(np.argmax(normalized))
            if normalized[lag - min_lag] >= 0.30:
                f0 = sample_rate / lag
        times.append((start + frame_size / 2) / sample_rate)
        values.append(f0)
    return np.asarray(times, dtype=np.float64), np.asarray(values, dtype=np.float64)


def _slice(waveform: np.ndarray, sample_rate: int, start_s: float, end_s: float) -> np.ndarray:
    start = max(0, round(start_s * sample_rate))
    end = min(len(waveform), round(end_s * sample_rate))
    return waveform[start:end]


def _rms(samples: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(samples, dtype=np.float64)))) if len(samples) else 0.0


def _f0_slope(times: np.ndarray, values: np.ndarray, interval: VowelInterval) -> float | None:
    mask = (times >= interval.start_s) & (times <= interval.end_s) & np.isfinite(values)
    if int(np.sum(mask)) < 2:
        return None
    return float(np.polyfit(times[mask], values[mask], deg=1)[0])
