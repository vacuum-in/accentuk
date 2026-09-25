from pathlib import Path
from typing import Any

import numpy as np

from ukstress.alignment import WhisperXWordAligner
from ukstress.alignment.interface import WordAlignmentResult
from ukstress.alignment.validation import (
    find_target_alignment,
    validate_word_alignments,
)
from ukstress.asr import ASRSegment
from ukstress.config import load_config
from ukstress.config.models import AlignmentConfig
from ukstress.datasets import WordAlignment
from ukstress.provenance import ExperimentManifest


class MockWhisperX:
    def __init__(self) -> None:
        self.load_calls: list[dict[str, Any]] = []
        self.align_calls: list[dict[str, Any]] = []

    def load_align_model(self, **kwargs: Any) -> tuple[object, dict[str, Any]]:
        self.load_calls.append(kwargs)
        return object(), {"language": "uk", "type": "huggingface", "dictionary": {}}

    def align(self, segments: list[dict[str, Any]], *_args: Any, **kwargs: Any) -> dict[str, Any]:
        self.align_calls.append({"segments": segments, **kwargs})
        return {
            "segments": [
                {
                    "words": [
                        {"word": "працівник", "start": 0.0, "end": 0.35, "score": 0.96},
                        {"word": "перевірив", "start": 0.36, "end": 0.7, "score": 0.94},
                        {"word": "замок", "start": 0.72, "end": 1.0, "score": 0.98},
                    ]
                }
            ]
        }


def _whisperx_config(**updates: Any) -> AlignmentConfig:
    return AlignmentConfig(
        word_backend="whisperx",
        word_model_id="Yehor/wav2vec2-xls-r-300m-uk-with-small-lm",
        **updates,
    )


def test_whisperx_backend_loads_explicit_ukrainian_ctc_model_and_normalizes_output() -> None:
    module = MockWhisperX()
    config = _whisperx_config(device="cpu")
    backend = WhisperXWordAligner(config, whisperx_module=module)

    result = backend.align(
        np.zeros(16_000, dtype=np.float32),
        16_000,
        "Працівник перевірив замок.",
    )

    assert module.load_calls[0]["language_code"] == "uk"
    assert module.load_calls[0]["model_name"] == config.word_model_id
    assert module.align_calls[0]["return_char_alignments"] is True
    assert [word.normalized_token for word in result.words] == [
        "працівник",
        "перевірив",
        "замок",
    ]
    assert result.words[-1].char_start == 20
    assert result.quality > 0.95
    assert result.words[-1].backend == "whisperx"
    assert result.words[-1].model_id == config.word_model_id


def test_whisperx_backend_aligns_timestamped_asr_segments() -> None:
    module = MockWhisperX()
    backend = WhisperXWordAligner(_whisperx_config(device="cpu"), whisperx_module=module)

    backend.align_asr_segments(
        np.zeros(16_000, dtype=np.float32),
        16_000,
        [
            ASRSegment(text="Працівник перевірив", start_s=0.0, end_s=0.7),
            ASRSegment(text="замок", start_s=0.72, end_s=1.0),
        ],
    )

    assert len(module.align_calls[0]["segments"]) == 2
    assert module.align_calls[0]["segments"][1]["start"] == 0.72


def test_alignment_validation_checks_quality_bounds_identity_and_target() -> None:
    backend = WhisperXWordAligner(_whisperx_config(), whisperx_module=MockWhisperX())
    result = backend.align(
        np.zeros(16_000, dtype=np.float32), 16_000, "працівник перевірив замок"
    )

    validation = validate_word_alignments(
        result,
        "працівник перевірив замок",
        min_quality=0.8,
    )

    assert validation.valid
    assert validation.identity_score == 1.0
    target = find_target_alignment(result, "ЗА́МОК")
    assert target is not None
    assert target.start_s == 0.72
    assert find_target_alignment(result, "замовк") is None


def test_alignment_validation_rejects_overlap_bounds_and_identity_mismatch() -> None:
    result = WordAlignmentResult(
        backend="whisperx",
        model_id="model",
        utterance_duration_s=1.0,
        words=[
            WordAlignment(
                token="це",
                normalized_token="це",
                start_s=0.0,
                end_s=0.7,
                confidence=0.9,
                char_start=0,
                char_end=2,
                backend="whisperx",
                model_id="model",
            ),
            WordAlignment(
                token="замовк",
                normalized_token="замовк",
                start_s=0.6,
                end_s=1.1,
                confidence=0.2,
                char_start=3,
                char_end=8,
                backend="whisperx",
                model_id="model",
            ),
        ],
    )

    validation = validate_word_alignments(result, "це замок", min_quality=0.8)

    assert not validation.valid
    assert set(validation.reasons) == {
        "alignment_out_of_bounds",
        "alignment_overlap",
        "target_identity_mismatch",
        "low_alignment_quality",
    }


def test_alignment_configuration_and_manifest_preserve_backend_versions(tmp_path: Path) -> None:
    config_path = Path("configs/alignment/whisperx-uk.yaml")
    config = load_config(config_path)
    manifest = ExperimentManifest(
        run_id="alignment-run",
        command=["ukstress", "align"],
        config_snapshot=config.model_dump(mode="json"),
        config_fingerprint="sha256:config",
        random_seeds={"python": 17},
        inputs={"corpus": "sha256:corpus"},
        alignment={
            "word_backend": config.alignment.word_backend,
            "word_model_id": config.alignment.word_model_id,
        },
    )
    path = tmp_path / "manifest.yaml"
    manifest.write(path)

    restored = ExperimentManifest.read(path)
    assert restored.alignment["word_backend"] == "whisperx"
    assert restored.alignment["word_model_id"] == config.alignment.word_model_id
