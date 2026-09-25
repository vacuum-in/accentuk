"""Ukrainian WhisperX CTC forced word alignment."""

from __future__ import annotations

import importlib
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

from ukstress.alignment.interface import WordAlignmentResult
from ukstress.alignment.validation import alignment_quality
from ukstress.asr.interface import ASRSegment
from ukstress.config.models import AlignmentConfig
from ukstress.datasets import WordAlignment
from ukstress.text import normalize_transcript


class WhisperXWordAligner:
    """Align a trusted Ukrainian transcript using a language-specific CTC model."""

    def __init__(
        self,
        config: AlignmentConfig,
        *,
        whisperx_module: Any | None = None,
        align_model: Any | None = None,
        align_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if config.word_backend != "whisperx":
            raise ValueError("WhisperXWordAligner requires alignment.word_backend='whisperx'")
        if config.language != "uk":
            raise ValueError("WhisperXWordAligner currently requires Ukrainian language code 'uk'")
        self.config = config
        self._whisperx = whisperx_module
        self._align_model = align_model
        self._align_metadata = dict(align_metadata) if align_metadata is not None else None

    @property
    def backend_id(self) -> str:
        return "whisperx"

    @property
    def model_id(self) -> str:
        return self.config.word_model_id

    def _load_backend(self) -> tuple[Any, Any, Mapping[str, Any]]:
        if self._whisperx is None:
            try:
                self._whisperx = importlib.import_module("whisperx")
            except ImportError as error:
                raise RuntimeError(
                    "WhisperX alignment is optional; install it with `uv sync --extra alignment`"
                ) from error
        if self._align_model is None or self._align_metadata is None:
            model, metadata = self._whisperx.load_align_model(
                language_code=self.config.language,
                device=self.config.device,
                model_name=self.config.word_model_id,
                model_dir=(
                    str(self.config.model_cache_dir)
                    if self.config.model_cache_dir is not None
                    else None
                ),
                model_cache_only=self.config.model_cache_only,
            )
            self._align_model = model
            self._align_metadata = metadata
        return self._whisperx, self._align_model, self._align_metadata

    def align(
        self,
        waveform: np.ndarray,
        sample_rate: int,
        normalized_transcript: str,
    ) -> WordAlignmentResult:
        if waveform.ndim != 1 or sample_rate != 16_000:
            raise ValueError("WhisperX alignment requires a mono 16-kHz waveform")
        transcript = normalize_transcript(normalized_transcript)
        if not transcript:
            raise ValueError("alignment transcript must contain at least one spoken token")
        duration_s = len(waveform) / sample_rate
        whisperx, model, metadata = self._load_backend()
        output = whisperx.align(
            [{"start": 0.0, "end": duration_s, "text": transcript}],
            model,
            metadata,
            waveform,
            self.config.device,
            return_char_alignments=self.config.return_char_alignments,
            print_progress=False,
        )
        words = normalize_whisperx_output(
            output,
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

    def align_with_chars(
        self,
        waveform: np.ndarray,
        sample_rate: int,
        normalized_transcript: str,
    ) -> tuple[WordAlignmentResult, Mapping[str, Any]]:
        """Word alignment plus the raw output, so the characters are not thrown away.

        ``align`` already asks WhisperX for character timings and then keeps
        only the words. Running the fine aligner afterwards re-runs the model
        once per word — 121 forward passes over a window where one suffices.
        Returning the raw output lets a caller take both from the single pass.
        """
        if waveform.ndim != 1 or sample_rate != 16_000:
            raise ValueError("WhisperX alignment requires a mono 16-kHz waveform")
        transcript = normalize_transcript(normalized_transcript)
        if not transcript:
            raise ValueError("alignment transcript must contain at least one spoken token")
        duration_s = len(waveform) / sample_rate
        whisperx, model, metadata = self._load_backend()
        output = whisperx.align(
            [{"start": 0.0, "end": duration_s, "text": transcript}],
            model,
            metadata,
            waveform,
            self.config.device,
            return_char_alignments=True,
            print_progress=False,
        )
        words = normalize_whisperx_output(
            output,
            transcript=transcript,
            backend=self.backend_id,
            model_id=self.model_id,
        )
        quality, _, _ = alignment_quality(words, transcript)
        return (
            WordAlignmentResult(
                backend=self.backend_id,
                model_id=self.model_id,
                utterance_duration_s=duration_s,
                words=words,
                quality=quality,
            ),
            output,
        )

    def align_asr_segments(
        self,
        waveform: np.ndarray,
        sample_rate: int,
        segments: list[ASRSegment],
    ) -> WordAlignmentResult:
        """Align timestamped ASR segments without constructing one full-file CTC trellis."""

        if waveform.ndim != 1 or sample_rate != 16_000:
            raise ValueError("WhisperX alignment requires a mono 16-kHz waveform")
        normalized_segments = [
            {"start": segment.start_s, "end": segment.end_s, "text": text}
            for segment in segments
            if (text := normalize_transcript(segment.text))
        ]
        if not normalized_segments:
            raise ValueError("alignment requires at least one non-empty ASR segment")
        transcript = " ".join(str(segment["text"]) for segment in normalized_segments)
        duration_s = len(waveform) / sample_rate
        whisperx, model, metadata = self._load_backend()
        output = whisperx.align(
            normalized_segments,
            model,
            metadata,
            np.ascontiguousarray(waveform).copy(),
            self.config.device,
            return_char_alignments=self.config.return_char_alignments,
            print_progress=False,
        )
        words = normalize_whisperx_output(
            output,
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


def _raw_words(output: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    word_segments = output.get("word_segments")
    if isinstance(word_segments, list):
        return (word for word in word_segments if isinstance(word, Mapping))
    segments = output.get("segments", [])
    return (
        word
        for segment in segments
        if isinstance(segment, Mapping)
        for word in segment.get("words", [])
        if isinstance(word, Mapping)
    )


def normalize_whisperx_output(
    output: Mapping[str, Any],
    *,
    transcript: str,
    backend: str,
    model_id: str,
) -> list[WordAlignment]:
    """Convert WhisperX mappings into canonical, ordered word records."""

    words: list[WordAlignment] = []
    cursor = 0
    for raw_word in _raw_words(output):
        token = str(raw_word.get("word", "")).strip()
        normalized_token = normalize_transcript(token)
        start = raw_word.get("start")
        end = raw_word.get("end")
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
        raw_score = raw_word.get("score", raw_word.get("confidence"))
        confidence = None if raw_score is None else max(0.0, min(1.0, float(raw_score)))
        words.append(
            WordAlignment(
                token=token,
                normalized_token=normalized_token,
                start_s=float(start),
                end_s=float(end),
                confidence=confidence,
                char_start=char_start,
                char_end=char_end,
                backend=backend,
                model_id=model_id,
            )
        )
    return words
