"""Learnable attention pooling over SSL frames."""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np

from ukstress.datasets import VowelInterval
from ukstress.features.ssl import time_to_frame_bounds


class LearnableAttentionPooler:
    """Small torch attention module loaded only by the optional feature path."""

    def __init__(self, feature_size: int) -> None:
        if feature_size < 1:
            raise ValueError("feature_size must be positive")
        try:
            torch = importlib.import_module("torch")
        except ImportError as error:
            raise RuntimeError("attention pooling requires `uv sync --extra features`") from error
        self._torch: Any = torch
        self._module: Any = torch.nn.Sequential(
            torch.nn.Linear(feature_size, feature_size),
            torch.nn.Tanh(),
            torch.nn.Linear(feature_size, 1, bias=False),
        )

    def parameters(self) -> Any:
        """Expose differentiable parameters to the future ranker optimizer."""

        return self._module.parameters()

    def pool(self, frames: np.ndarray) -> np.ndarray:
        """Return one attention-weighted vector for a non-empty frame matrix."""

        if frames.ndim != 2 or not frames.shape[0]:
            raise ValueError("frames must be a non-empty [frames, features] matrix")
        values = self._torch.as_tensor(frames, dtype=self._torch.float32)
        logits = self._module(values).squeeze(-1)
        weights = self._torch.softmax(logits, dim=0)
        pooled = (weights.unsqueeze(-1) * values).sum(dim=0).detach().cpu().numpy()
        return np.asarray(pooled, dtype=np.float32)


def attention_pool_intervals(
    frame_embeddings: np.ndarray,
    intervals: list[VowelInterval],
    pooler: LearnableAttentionPooler,
    *,
    frame_shift_s: float,
    context_ms: int = 40,
) -> list[list[float]]:
    """Attention-pool every vowel after expanding its temporal context."""

    if context_ms < 0:
        raise ValueError("context_ms must be non-negative")
    if frame_embeddings.ndim != 2 or not frame_embeddings.shape[0]:
        raise ValueError("frame_embeddings must be a non-empty [frames, features] matrix")
    padding_s = context_ms / 1000.0
    pooled: list[list[float]] = []
    for interval in intervals:
        start, end = time_to_frame_bounds(
            max(0.0, interval.start_s - padding_s),
            interval.end_s + padding_s,
            frame_shift_s=frame_shift_s,
            frame_count=frame_embeddings.shape[0],
        )
        pooled.append([float(value) for value in pooler.pool(frame_embeddings[start:end]).tolist()])
    return pooled
