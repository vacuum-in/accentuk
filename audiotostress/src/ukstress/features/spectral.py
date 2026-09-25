"""Optional spectral and formant feature extension points."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from ukstress.datasets import VowelInterval


@runtime_checkable
class SpectralFeatureHook(Protocol):
    """Optional extractor; callers may omit it without changing core features."""

    @property
    def hook_id(self) -> str: ...

    def extract(
        self, waveform: np.ndarray, sample_rate: int, interval: VowelInterval
    ) -> dict[str, float | bool | None]: ...


def extract_optional_spectral_features(
    hook: SpectralFeatureHook | None,
    waveform: np.ndarray,
    sample_rate: int,
    intervals: list[VowelInterval],
) -> list[dict[str, float | bool | None]]:
    """Return one mapping per vowel, or empty mappings when no hook is configured."""

    if hook is None:
        return [{} for _ in intervals]
    return [hook.extract(waveform, sample_rate, interval) for interval in intervals]
