"""Lazy adapter for faster-whisper and compatible mocked results."""

from __future__ import annotations

import importlib
import math
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

from ukstress.asr.interface import ASRResult, ASRSegment, ASRToken
from ukstress.config.models import ASRConfig


class WhisperCompatibleBackend:
    def __init__(self, config: ASRConfig, *, model: Any | None = None) -> None:
        self.config = config
        self._model = model

    @property
    def backend_id(self) -> str:
        return self.config.backend

    @property
    def model_id(self) -> str:
        return self.config.model_id

    def _get_model(self) -> Any:
        if self._model is None:
            if self.config.backend != "faster-whisper":
                raise ValueError(f"unsupported Whisper backend {self.config.backend!r}")
            try:
                module = importlib.import_module("faster_whisper")
            except ImportError as error:
                raise RuntimeError(
                    "faster-whisper is not installed; install the ASR extra or inject a "
                    "compatible model"
                ) from error
            self._model = module.WhisperModel(
                self.config.model_id,
                device=self.config.device,
                compute_type=self.config.compute_type,
            )
        return self._model

    def transcribe(self, waveform: np.ndarray, sample_rate: int) -> ASRResult:
        if waveform.ndim != 1 or sample_rate <= 0:
            raise ValueError("ASR expects a mono waveform and positive sample rate")
        if sample_rate != 16_000:
            raise ValueError("Whisper-compatible backend expects canonical 16-kHz audio")
        response = self._get_model().transcribe(
            waveform,
            language=self.config.language,
            beam_size=self.config.beam_size,
            word_timestamps=True,
        )
        raw_segments = response[0] if isinstance(response, tuple) else response.get("segments", [])
        segments = [
            normalized
            for segment in raw_segments
            if (normalized := _normalize_segment(segment)) is not None
        ]
        transcript = " ".join(segment.text.strip() for segment in segments if segment.text.strip())
        return ASRResult(
            transcript=transcript,
            language=self.config.language,
            backend=self.backend_id,
            model_id=self.model_id,
            segments=segments,
        )


def _value(item: Any, name: str, default: Any = None) -> Any:
    if isinstance(item, Mapping):
        return item.get(name, default)
    return getattr(item, name, default)


def _probability(value: Any, *, log_probability: bool = False) -> float | None:
    if value is None:
        return None
    probability = math.exp(float(value)) if log_probability else float(value)
    return max(0.0, min(1.0, probability))


def _normalize_token(raw: Any) -> ASRToken | None:
    start = _value(raw, "start")
    end = _value(raw, "end")
    if (start is None) != (end is None):
        return None
    if start is not None and end is not None and float(start) >= float(end):
        return None
    return ASRToken(
        text=str(_value(raw, "word", _value(raw, "text", ""))),
        start_s=start,
        end_s=end,
        confidence=_probability(_value(raw, "probability", _value(raw, "confidence"))),
    )


def _normalize_segment(raw: Any) -> ASRSegment | None:
    raw_words: Iterable[Any] = _value(raw, "words", []) or []
    tokens = [token for word in raw_words if (token := _normalize_token(word)) is not None]
    start = float(_value(raw, "start"))
    end = float(_value(raw, "end"))
    if start >= end:
        return None
    confidence = _value(raw, "confidence")
    if confidence is None:
        confidence = _probability(_value(raw, "avg_logprob"), log_probability=True)
    else:
        confidence = _probability(confidence)
    return ASRSegment(
        text=str(_value(raw, "text", "")),
        start_s=start,
        end_s=end,
        confidence=confidence,
        tokens=tokens,
    )
