from dataclasses import dataclass

import numpy as np
import pytest

from ukstress.asr import WhisperCompatibleBackend
from ukstress.asr.crosscheck import target_identity_matches, transcript_identity_score
from ukstress.config.models import ASRConfig


@dataclass
class MockWord:
    word: str
    start: float
    end: float
    probability: float


@dataclass
class MockSegment:
    text: str
    start: float
    end: float
    avg_logprob: float
    words: list[MockWord]


class MockWhisper:
    def transcribe(
        self, waveform: np.ndarray, **options: object
    ) -> tuple[list[MockSegment], object]:
        assert waveform.shape == (16_000,)
        assert options["language"] == "uk"
        return (
            [
                MockSegment(
                    text="Це замок.",
                    start=0.0,
                    end=1.0,
                    avg_logprob=-0.1,
                    words=[MockWord(" замок", 0.4, 0.9, 0.97)],
                )
            ],
            object(),
        )


def test_whisper_adapter_normalizes_mocked_segments_and_confidence() -> None:
    backend = WhisperCompatibleBackend(
        ASRConfig(model_id="mock-v1"), model=MockWhisper()
    )

    result = backend.transcribe(np.zeros(16_000, dtype=np.float32), 16_000)

    assert result.transcript == "Це замок."
    assert result.backend == "faster-whisper"
    assert result.model_id == "mock-v1"
    assert result.segments[0].confidence == pytest.approx(0.9048, abs=0.001)
    assert result.segments[0].tokens[0].confidence == 0.97


def test_whisper_adapter_accepts_mapping_style_results() -> None:
    class MappingWhisper:
        def transcribe(self, *_args: object, **_kwargs: object) -> dict[str, object]:
            return {
                "segments": [
                    {
                        "text": "Замок",
                        "start": 0.0,
                        "end": 0.5,
                        "confidence": 0.8,
                        "words": [],
                    }
                ]
            }

    backend = WhisperCompatibleBackend(ASRConfig(), model=MappingWhisper())

    result = backend.transcribe(np.zeros(16_000, dtype=np.float32), 16_000)
    assert result.segments[0].confidence == 0.8


def test_whisper_adapter_discards_zero_duration_tokens() -> None:
    class MappingWhisper:
        def transcribe(self, *_args: object, **_kwargs: object) -> dict[str, object]:
            return {
                "segments": [
                    {
                        "text": "й замок",
                        "start": 0.0,
                        "end": 0.5,
                        "words": [
                            {"word": " й", "start": 0.1, "end": 0.1},
                            {"word": " замок", "start": 0.1, "end": 0.5},
                        ],
                    }
                ]
            }

    result = WhisperCompatibleBackend(ASRConfig(), model=MappingWhisper()).transcribe(
        np.zeros(16_000, dtype=np.float32), 16_000
    )

    assert [token.text.strip() for token in result.segments[0].tokens] == ["замок"]


def test_reference_crosscheck_is_deterministic_and_stress_insensitive() -> None:
    assert transcript_identity_score("Це за́мок", "це замок") == 1.0
    assert transcript_identity_score("це старий замок", "це замок") == pytest.approx(2 / 3)
    assert target_identity_matches("ЗА́МОК", "замок")
    assert not target_identity_matches("замовк", "замок")


def test_whisper_adapter_requires_canonical_sample_rate() -> None:
    backend = WhisperCompatibleBackend(ASRConfig(), model=MockWhisper())

    with pytest.raises(ValueError, match="16-kHz"):
        backend.transcribe(np.zeros(8_000, dtype=np.float32), 8_000)
