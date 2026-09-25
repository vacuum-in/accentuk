"""Canary auxiliary CTC forced alignment through the CrispASR GGUF runtime."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from ukstress.alignment.interface import WordAlignmentResult
from ukstress.alignment.validation import alignment_quality
from ukstress.config.models import AlignmentConfig
from ukstress.datasets import WordAlignment
from ukstress.text import normalize_transcript

DownloadFunction = Callable[..., str]
AlignFunction = Callable[..., list[Any]]


class CanaryCTCWordAligner:
    """Force a Ukrainian transcript against Canary's standalone CTC timestamp model."""

    def __init__(
        self,
        config: AlignmentConfig,
        *,
        align_function: AlignFunction | None = None,
        download_function: DownloadFunction | None = None,
    ) -> None:
        if config.word_backend != "canary_ctc":
            raise ValueError("CanaryCTCWordAligner requires word_backend='canary_ctc'")
        if config.language != "uk":
            raise ValueError("CanaryCTCWordAligner currently requires Ukrainian language code 'uk'")
        self.config = config
        self._align_function = align_function
        self._download_function = download_function
        self._resolved_model_path: Path | None = None

    @property
    def backend_id(self) -> str:
        return "canary_ctc_gguf"

    @property
    def model_id(self) -> str:
        return (
            f"{self.config.word_model_id}@{self.config.model_revision}:{self.config.model_filename}"
        )

    def _get_align_function(self) -> AlignFunction:
        if self._align_function is None:
            try:
                module = importlib.import_module("crispasr")
            except ImportError as error:
                raise RuntimeError(
                    "Canary CTC alignment is optional; install it with "
                    "`uv sync --extra canary-alignment`"
                ) from error
            self._align_function = module.align_words
        return self._align_function

    def _get_download_function(self) -> DownloadFunction:
        if self._download_function is None:
            try:
                module = importlib.import_module("huggingface_hub")
            except ImportError as error:
                raise RuntimeError(
                    "huggingface-hub is required to resolve the Canary CTC model"
                ) from error
            self._download_function = module.hf_hub_download
        return self._download_function

    def _resolve_model_path(self) -> Path:
        if self._resolved_model_path is not None:
            return self._resolved_model_path
        if self.config.model_path is not None:
            path = self.config.model_path
            if not path.is_file():
                raise FileNotFoundError(f"Canary CTC model does not exist: {path}")
        else:
            downloaded = self._get_download_function()(
                repo_id=self.config.word_model_id,
                filename=self.config.model_filename,
                revision=self.config.model_revision,
                cache_dir=(
                    str(self.config.model_cache_dir)
                    if self.config.model_cache_dir is not None
                    else None
                ),
                local_files_only=self.config.model_cache_only,
            )
            path = Path(downloaded)
        self._resolved_model_path = path
        return path

    def align(
        self,
        waveform: np.ndarray,
        sample_rate: int,
        normalized_transcript: str,
    ) -> WordAlignmentResult:
        if waveform.ndim != 1 or sample_rate != 16_000:
            raise ValueError("Canary CTC alignment requires a mono 16-kHz waveform")
        transcript = normalize_transcript(normalized_transcript)
        if not transcript:
            raise ValueError("alignment transcript must contain at least one spoken token")
        duration_s = len(waveform) / sample_rate
        raw_words = self._get_align_function()(
            str(self._resolve_model_path()),
            transcript,
            np.ascontiguousarray(waveform, dtype=np.float32),
            n_threads=self.config.n_threads,
        )
        words = normalize_canary_ctc_output(
            raw_words,
            transcript=transcript,
            backend=self.backend_id,
            model_id=self.model_id,
        )
        quality, _, _ = alignment_quality(words, transcript)
        return WordAlignmentResult(
            backend=self.backend_id,
            model_id=self.model_id,
            utterance_duration_s=duration_s,
            words=words,
            quality=quality,
        )


def _value(item: Any, name: str) -> Any:
    if isinstance(item, Mapping):
        return item.get(name)
    return getattr(item, name, None)


def normalize_canary_ctc_output(
    raw_words: list[Any],
    *,
    transcript: str,
    backend: str,
    model_id: str,
) -> list[WordAlignment]:
    """Convert CrispASR AlignedWord objects into canonical records."""

    words: list[WordAlignment] = []
    cursor = 0
    for raw_word in raw_words:
        token = str(_value(raw_word, "text") or "").strip()
        normalized_token = normalize_transcript(token)
        start = _value(raw_word, "start")
        end = _value(raw_word, "end")
        if not normalized_token or start is None or end is None:
            continue
        found_start = transcript.find(normalized_token, cursor)
        char_start: int | None
        char_end: int | None
        if found_start < 0:
            char_start = None
            char_end = None
        else:
            char_start = found_start
            char_end = char_start + len(normalized_token)
            cursor = char_end
        words.append(
            WordAlignment(
                token=token,
                normalized_token=normalized_token,
                start_s=float(start),
                end_s=float(end),
                confidence=None,
                char_start=char_start,
                char_end=char_end,
                backend=backend,
                model_id=model_id,
            )
        )
    return words
