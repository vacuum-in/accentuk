from typing import Any

import numpy as np

from ukstress.alignment import WhisperXCharFineAligner
from ukstress.config.models import AlignmentConfig
from ukstress.datasets import WordAlignment


class MockWhisperXChars:
    def load_align_model(self, **_kwargs: Any) -> tuple[object, dict[str, Any]]:
        return object(), {"language": "uk", "type": "huggingface", "dictionary": {}}

    def align(self, _segments: list[dict[str, Any]], *_args: Any, **kwargs: Any) -> dict[str, Any]:
        assert kwargs["return_char_alignments"] is True
        return {
            "segments": [
                {
                    "chars": [
                        {"char": "з", "start": 0.10, "end": 0.15, "score": 0.99},
                        {"char": "а", "start": 0.15, "end": 0.22, "score": 0.98},
                        {"char": "м", "start": 0.22, "end": 0.30, "score": 0.97},
                        {"char": "о", "start": 0.30, "end": 0.37, "score": 0.96},
                        {"char": "к", "start": 0.37, "end": 0.45, "score": 0.95},
                    ]
                }
            ]
        }


def test_whisperx_character_aligner_emits_timed_ukrainian_vowels() -> None:
    config = AlignmentConfig(
        word_backend="whisperx",
        word_model_id="Yehor/wav2vec2-xls-r-300m-uk-with-small-lm",
        device="cpu",
        fine_backend="whisperx_chars",
    )
    aligner = WhisperXCharFineAligner(config, whisperx_module=MockWhisperXChars())
    word = WordAlignment(
        token="замок",
        normalized_token="замок",
        start_s=0.10,
        end_s=0.45,
        backend="whisperx",
    )

    result = aligner.align(np.zeros(16_000, dtype=np.float32), 16_000, word, "замок")

    assert result.backend == "whisperx_char_ctc"
    assert result.model_id == config.word_model_id
    assert [(v.grapheme, v.start_s, v.end_s) for v in result.vowels] == [
        ("а", 0.15, 0.22),
        ("о", 0.30, 0.37),
    ]
    assert result.quality == 0.97
    assert result.rejection_reasons == []
