"""Deterministic duration batching and optional mixed-precision context."""

from __future__ import annotations

import contextlib
import importlib
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DurationBatch:
    indices: tuple[int, ...]
    total_duration_s: float


def batch_indices_by_duration(
    durations_s: Sequence[float], *, max_batch_duration_s: float
) -> list[DurationBatch]:
    """Stable length-aware batches bounded by total source duration."""

    if max_batch_duration_s <= 0.0:
        raise ValueError("max_batch_duration_s must be positive")
    if any(duration <= 0.0 for duration in durations_s):
        raise ValueError("all item durations must be positive")
    batches: list[DurationBatch] = []
    current: list[int] = []
    total = 0.0
    for index, duration in sorted(enumerate(durations_s), key=lambda item: (item[1], item[0])):
        if current and total + duration > max_batch_duration_s:
            batches.append(DurationBatch(indices=tuple(current), total_duration_s=total))
            current, total = [], 0.0
        current.append(index)
        total += duration
    if current:
        batches.append(DurationBatch(indices=tuple(current), total_duration_s=total))
    return batches


def batch_for_device(
    durations_s: Sequence[float], *, max_batch_duration_s: float, device: str
) -> list[DurationBatch]:
    """Duration-batch SSL work for CPU/GPU with the same deterministic ordering."""

    if not device.strip():
        raise ValueError("device must be non-empty")
    return batch_indices_by_duration(durations_s, max_batch_duration_s=max_batch_duration_s)


@contextlib.contextmanager
def inference_autocast(device: str, *, mixed_precision: bool) -> Iterator[None]:
    """Enable torch autocast only for CUDA inference when explicitly configured."""

    if not mixed_precision:
        yield
        return
    if not device.startswith("cuda"):
        raise ValueError("mixed_precision is supported only for CUDA devices")
    try:
        torch: Any = importlib.import_module("torch")
    except ImportError as error:
        raise RuntimeError("mixed precision requires `uv sync --extra features`") from error
    with torch.autocast(device_type="cuda"):
        yield
