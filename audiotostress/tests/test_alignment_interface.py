import numpy as np

from ukstress.alignment import WordAlignmentBackend, WordAlignmentResult
from ukstress.datasets import WordAlignment


class MockAligner:
    @property
    def backend_id(self) -> str:
        return "mock-ctc"

    @property
    def model_id(self) -> str:
        return "mock-v1"

    def align(
        self,
        waveform: np.ndarray,
        sample_rate: int,
        normalized_transcript: str,
    ) -> WordAlignmentResult:
        assert waveform.ndim == 1
        assert sample_rate == 16_000
        assert normalized_transcript == "це замок"
        return WordAlignmentResult(
            backend=self.backend_id,
            model_id=self.model_id,
            utterance_duration_s=len(waveform) / sample_rate,
            words=[
                WordAlignment(
                    token="замок",
                    normalized_token="замок",
                    start_s=0.4,
                    end_s=0.9,
                    confidence=0.95,
                    backend=self.backend_id,
                    model_id=self.model_id,
                )
            ],
        )


def test_mock_backend_satisfies_backend_neutral_contract() -> None:
    backend = MockAligner()

    assert isinstance(backend, WordAlignmentBackend)
    result = backend.align(np.zeros(16_000, dtype=np.float32), 16_000, "це замок")
    assert result.words[0].normalized_token == "замок"
    assert result.words[0].backend == result.backend
