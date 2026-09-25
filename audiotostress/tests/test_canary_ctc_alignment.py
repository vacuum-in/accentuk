from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ukstress.alignment import CanaryCTCWordAligner
from ukstress.alignment.validation import find_target_alignment, validate_word_alignments
from ukstress.config import load_config
from ukstress.config.models import AlignmentConfig


@dataclass
class MockAlignedWord:
    text: str
    start: float
    end: float


def _canary_config(**updates: Any) -> AlignmentConfig:
    return AlignmentConfig(
        word_backend="canary_ctc",
        word_model_id="cstr/canary-ctc-aligner-GGUF",
        **updates,
    )


def test_canary_ctc_backend_pins_model_and_normalizes_crispasr_output(tmp_path: Path) -> None:
    model_path = tmp_path / "canary-ctc-aligner-q8_0.gguf"
    model_path.touch()
    download_calls: list[dict[str, Any]] = []
    align_calls: list[tuple[str, str, int]] = []

    def download(**kwargs: Any) -> str:
        download_calls.append(kwargs)
        return str(model_path)

    def align(model: str, transcript: str, pcm: np.ndarray, *, n_threads: int) -> list[Any]:
        assert pcm.dtype == np.float32
        align_calls.append((model, transcript, n_threads))
        return [
            MockAlignedWord("працівник", 0.00, 0.32),
            MockAlignedWord("перевірив", 0.40, 0.72),
            MockAlignedWord("замок", 0.80, 1.04),
        ]

    config = _canary_config(n_threads=6)
    backend = CanaryCTCWordAligner(
        config,
        align_function=align,
        download_function=download,
    )

    result = backend.align(
        np.zeros(20_000, dtype=np.float32), 16_000, "Працівник перевірив замок."
    )

    assert download_calls[0]["repo_id"] == "cstr/canary-ctc-aligner-GGUF"
    assert download_calls[0]["filename"] == "canary-ctc-aligner-q8_0.gguf"
    assert download_calls[0]["revision"] == "2b22dc9aff585bc8368a5228173c8afe22c155b7"
    assert align_calls == [(str(model_path), "працівник перевірив замок", 6)]
    assert result.backend == "canary_ctc_gguf"
    assert result.model_id == backend.model_id
    assert result.quality == 1.0
    assert result.words[-1].char_start == 20


def test_canary_ctc_result_passes_identity_and_topology_validation(tmp_path: Path) -> None:
    model_path = tmp_path / "model.gguf"
    model_path.touch()

    def align(*_args: Any, **_kwargs: Any) -> list[Any]:
        return [
            {"text": "це", "start": 0.00, "end": 0.16},
            {"text": "замок", "start": 0.24, "end": 0.64},
        ]

    backend = CanaryCTCWordAligner(
        _canary_config(model_path=model_path), align_function=align
    )
    result = backend.align(np.zeros(16_000, dtype=np.float32), 16_000, "це замок")

    validation = validate_word_alignments(result, "це замок")
    assert validation.valid
    assert find_target_alignment(result, "за́мок") == result.words[1]


def test_canary_configuration_remains_available_but_is_not_the_project_default() -> None:
    default = AlignmentConfig()
    loaded = load_config("configs/alignment/canary-ctc-uk.yaml")

    assert default.word_backend == "whisperx"
    assert default.word_model_id == "Yehor/wav2vec2-xls-r-300m-uk-with-small-lm"
    assert loaded.alignment.word_backend == "canary_ctc"
    assert loaded.alignment.model_filename.endswith("q8_0.gguf")
