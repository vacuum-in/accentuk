"""Optional Hugging Face speech encoder and deterministic interval pooling."""

from __future__ import annotations

import importlib
from typing import Any, Protocol, runtime_checkable

import numpy as np

from ukstress.datasets import VowelInterval


@runtime_checkable
class SpeechEncoder(Protocol):
    """Minimal encoder contract used by feature extraction."""

    @property
    def model_id(self) -> str: ...

    @property
    def frame_shift_s(self) -> float: ...

    def encode(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray: ...


class HuggingFaceSpeechEncoder:
    """Load a Transformers speech encoder only when the optional feature path is used."""

    def __init__(
        self,
        model_id: str,
        *,
        device: str = "cpu",
        frame_shift_s: float = 0.02,
        model_cache_dir: str | None = None,
        model_cache_only: bool = False,
        processor: Any | None = None,
        model: Any | None = None,
    ) -> None:
        if not model_id.strip():
            raise ValueError("model_id must be non-empty")
        if frame_shift_s <= 0.0:
            raise ValueError("frame_shift_s must be positive")
        self._model_id = model_id
        self.device = device
        self._frame_shift_s = frame_shift_s
        self.model_cache_dir = model_cache_dir
        self.model_cache_only = model_cache_only
        self._processor = processor
        self._model = model

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def frame_shift_s(self) -> float:
        return self._frame_shift_s

    def _load(self) -> tuple[Any, Any, Any]:
        if self._processor is None or self._model is None:
            try:
                transformers = importlib.import_module("transformers")
            except ImportError as error:
                raise RuntimeError(
                    "SSL feature extraction is optional; install it with `uv sync --extra features`"
                ) from error
            kwargs: dict[str, Any] = {}
            if self.model_cache_dir is not None:
                kwargs["cache_dir"] = self.model_cache_dir
            kwargs["local_files_only"] = self.model_cache_only
            self._processor = transformers.AutoProcessor.from_pretrained(self.model_id, **kwargs)
            self._model = transformers.AutoModel.from_pretrained(self.model_id, **kwargs)
            self._model.to(self.device)
            self._model.eval()
        return self._processor, self._model, importlib.import_module("torch")

    def encode(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        if waveform.ndim != 1 or sample_rate != 16_000:
            raise ValueError("speech encoder requires a mono 16-kHz waveform")
        processor, model, torch = self._load()
        inputs = processor(
            np.ascontiguousarray(waveform, dtype=np.float32),
            sampling_rate=sample_rate,
            return_tensors="pt",
        )
        inputs = {name: value.to(self.device) for name, value in inputs.items()}
        with torch.inference_mode():
            output = model(**inputs)
        hidden = output.last_hidden_state[0].detach().cpu().numpy()
        if hidden.ndim != 2 or not hidden.shape[0]:
            raise ValueError("speech encoder returned no frame embeddings")
        return np.asarray(hidden, dtype=np.float32)


def time_to_frame_bounds(
    start_s: float,
    end_s: float,
    *,
    frame_shift_s: float,
    frame_count: int,
) -> tuple[int, int]:
    """Map an interval to a clipped half-open frame range."""

    if frame_shift_s <= 0.0:
        raise ValueError("frame_shift_s must be positive")
    if frame_count < 1:
        raise ValueError("frame_count must be positive")
    if start_s < 0.0 or start_s >= end_s:
        raise ValueError("interval must have non-negative ordered bounds")
    start = max(0, min(frame_count - 1, int(np.floor(start_s / frame_shift_s))))
    end = max(start + 1, min(frame_count, int(np.ceil(end_s / frame_shift_s))))
    return start, end


def mean_pool_intervals(
    frame_embeddings: np.ndarray,
    intervals: list[VowelInterval],
    *,
    frame_shift_s: float,
) -> list[list[float]]:
    """Mean-pool encoder frames for each vowel, preserving interval order."""

    if frame_embeddings.ndim != 2 or not frame_embeddings.shape[0]:
        raise ValueError("frame_embeddings must be a non-empty [frames, features] matrix")
    pooled: list[list[float]] = []
    for interval in intervals:
        start, end = time_to_frame_bounds(
            interval.start_s,
            interval.end_s,
            frame_shift_s=frame_shift_s,
            frame_count=frame_embeddings.shape[0],
        )
        vector: Any = np.mean(frame_embeddings[start:end], axis=0, dtype=np.float64)
        pooled.append([float(value) for value in vector.tolist()])
    return pooled
