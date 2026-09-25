import numpy as np
import pytest

from ukstress.datasets import VowelInterval
from ukstress.features import HuggingFaceSpeechEncoder, mean_pool_intervals, time_to_frame_bounds


def test_time_to_frame_bounds_is_clipped_and_half_open() -> None:
    assert time_to_frame_bounds(0.021, 0.059, frame_shift_s=0.02, frame_count=4) == (1, 3)
    assert time_to_frame_bounds(0.0, 1.0, frame_shift_s=0.02, frame_count=4) == (0, 4)


def test_mean_pool_intervals_preserves_order_and_feature_width() -> None:
    embeddings = np.arange(20, dtype=np.float32).reshape(5, 4)
    intervals = [
        VowelInterval(vowel_index=0, grapheme="а", start_s=0.0, end_s=0.02),
        VowelInterval(vowel_index=1, grapheme="о", start_s=0.04, end_s=0.08),
    ]

    pooled = mean_pool_intervals(embeddings, intervals, frame_shift_s=0.02)

    assert pooled == [[0.0, 1.0, 2.0, 3.0], [10.0, 11.0, 12.0, 13.0]]


def test_huggingface_encoder_uses_injected_processor_and_model() -> None:
    class FakeTensor:
        def to(self, _device: str) -> "FakeTensor":
            return self

    class Processor:
        def __call__(self, *_args: object, **_kwargs: object) -> dict[str, object]:
            return {"input_values": FakeTensor()}

    class Output:
        last_hidden_state = FakeTensor()

    class Model:
        def to(self, _device: str) -> "Model":
            return self

        def eval(self) -> "Model":
            return self

        def __call__(self, **_kwargs: object) -> Output:
            return Output()

    # This test only checks constructor metadata; the real torch tensor path is optional.
    encoder = HuggingFaceSpeechEncoder("fixture/model", processor=Processor(), model=Model())
    assert encoder.model_id == "fixture/model"
    assert encoder.frame_shift_s == 0.02
    with pytest.raises(ValueError, match="positive"):
        HuggingFaceSpeechEncoder("fixture/model", frame_shift_s=0.0)
