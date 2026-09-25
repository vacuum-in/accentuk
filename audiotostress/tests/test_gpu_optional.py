import pytest


@pytest.mark.gpu
def test_optional_gpu_profile() -> None:
    """Opt-in smoke profile; default CI does not require CUDA or model downloads."""

    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable; run with -m gpu on a configured GPU worker")
    assert torch.cuda.device_count() >= 1
