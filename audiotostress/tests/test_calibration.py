import numpy as np
import pytest

from ukstress.calibration import IsotonicCalibrator, TemperatureCalibrator


def test_calibrators_require_held_out_manual_gold_and_emit_probabilities() -> None:
    logits = np.array([[3.0, 0.0], [0.0, 3.0]], dtype=np.float64)
    targets = np.array([0, 1])
    temperature = TemperatureCalibrator.fit(
        logits, targets, split_name="validation", manual_gold=True
    )
    isotonic = IsotonicCalibrator.fit(
        np.array([0.2, 0.8]), np.array([0, 1]), split_name="validation", manual_gold=True
    )

    assert temperature.transform(logits).shape == (2, 2)
    assert np.all(
        (isotonic.transform(np.array([0.1, 0.9])) >= 0.0)
        & (isotonic.transform(np.array([0.1, 0.9])) <= 1.0)
    )
    with pytest.raises(ValueError, match="held-out"):
        TemperatureCalibrator.fit(logits, targets, split_name="train", manual_gold=True)
