"""Character-CTC vowel alignment using WhisperX's Ukrainian Wav2Vec2 model."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

from ukstress.alignment.fine_interface import FineAlignmentResult
from ukstress.alignment.fine_validation import validate_vowel_intervals
from ukstress.alignment.whisperx_backend import WhisperXWordAligner
from ukstress.config.models import AlignmentConfig
from ukstress.datasets import VowelInterval, WordAlignment
from ukstress.lexicon.stress import UKRAINIAN_VOWELS
from ukstress.text import normalize_transcript


class WhisperXCharFineAligner:
    """Return timed Ukrainian vowel graphemes from WhisperX CTC character output.

    The underlying ``Yehor/wav2vec2-xls-r-300m-uk-with-small-lm`` checkpoint has a
    character-oriented CTC vocabulary.  WhisperX exposes its forced-alignment path as
    timed characters when ``return_char_alignments`` is enabled.  We align one known
    word inside its already aligned word interval, then retain only the vowel characters.
    """

    def __init__(
        self,
        config: AlignmentConfig,
        *,
        whisperx_module: Any | None = None,
        align_model: Any | None = None,
        align_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if config.word_backend != "whisperx":
            raise ValueError("WhisperXCharFineAligner requires alignment.word_backend='whisperx'")
        if not config.return_char_alignments:
            raise ValueError("WhisperXCharFineAligner requires return_char_alignments=true")
        self.config = config
        self._word_aligner = WhisperXWordAligner(
            config,
            whisperx_module=whisperx_module,
            align_model=align_model,
            align_metadata=align_metadata,
        )

    @property
    def backend_id(self) -> str:
        return "whisperx_char_ctc"

    @property
    def model_id(self) -> str:
        return self.config.word_model_id

    def align(
        self,
        waveform: np.ndarray,
        sample_rate: int,
        word: WordAlignment,
        normalized_word: str,
    ) -> FineAlignmentResult:
        if waveform.ndim != 1 or sample_rate != 16_000:
            raise ValueError("WhisperX character alignment requires a mono 16-kHz waveform")
        target = normalize_transcript(normalized_word)
        if not target or target != word.normalized_token:
            raise ValueError("normalized_word must match the word alignment token")
        if word.end_s > len(waveform) / sample_rate + 1e-6:
            raise ValueError("word alignment is outside the supplied waveform")

        whisperx, model, metadata = self._word_aligner._load_backend()
        output = whisperx.align(
            [{"start": word.start_s, "end": word.end_s, "text": target}],
            model,
            metadata,
            np.ascontiguousarray(waveform).copy(),
            self.config.device,
            return_char_alignments=True,
            print_progress=False,
        )
        vowels = _vowels_from_whisperx_chars(_raw_chars(output), target)
        validation = validate_vowel_intervals(
            vowels,
            target,
            word_start_s=word.start_s,
            word_end_s=word.end_s,
            utterance_duration_s=len(waveform) / sample_rate,
            overlap_tolerance_s=self.config.overlap_tolerance_s,
        )
        quality = _vowel_alignment_quality(vowels, target)
        return FineAlignmentResult(
            backend=self.backend_id,
            model_id=self.model_id,
            target_word=target,
            word_start_s=word.start_s,
            word_end_s=word.end_s,
            vowels=vowels,
            quality=quality,
        ).with_rejections(validation.reasons)


def _raw_chars(output: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    """Yield WhisperX character records across supported output layouts."""

    top_level = output.get("char_segments")
    if isinstance(top_level, list):
        yield from (char for char in top_level if isinstance(char, Mapping))
    for segment in output.get("segments", []):
        if not isinstance(segment, Mapping):
            continue
        chars = segment.get("chars", segment.get("char_segments", []))
        if isinstance(chars, list):
            yield from (char for char in chars if isinstance(char, Mapping))


def _vowels_from_whisperx_chars(
    raw_chars: Iterable[Mapping[str, Any]], target: str
) -> list[VowelInterval]:
    """Match monotonic CTC character spans to the vowel sequence in ``target``."""

    expected = [character for character in target if character in UKRAINIAN_VOWELS]
    vowels: list[VowelInterval] = []
    expected_cursor = 0
    for raw in raw_chars:
        if expected_cursor >= len(expected):
            break
        character = normalize_transcript(str(raw.get("char", "")))
        if len(character) != 1 or character != expected[expected_cursor]:
            continue
        start, end = raw.get("start"), raw.get("end")
        if start is None or end is None:
            continue
        start_s, end_s = float(start), float(end)
        if end_s <= start_s:
            continue
        score = raw.get("score", raw.get("confidence"))
        confidence = None if score is None else max(0.0, min(1.0, float(score)))
        vowels.append(
            VowelInterval(
                vowel_index=expected_cursor,
                grapheme=character,
                phone=character,
                start_s=start_s,
                end_s=end_s,
                confidence=confidence,
            )
        )
        expected_cursor += 1
    return vowels


def _vowel_alignment_quality(vowels: list[VowelInterval], target: str) -> float:
    expected_count = sum(character in UKRAINIAN_VOWELS for character in target)
    coverage = len(vowels) / expected_count if expected_count else 0.0
    confidences = [vowel.confidence for vowel in vowels if vowel.confidence is not None]
    mean_confidence = sum(confidences) / len(confidences) if confidences else 1.0
    return max(0.0, min(1.0, coverage * mean_confidence))
