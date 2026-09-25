import pytest

from ukstress.features import ProsodyNormalizer, extract_optional_spectral_features


def test_prosody_normalizer_is_train_only_persisted_and_keeps_boolean_masks(tmp_path) -> None:
    normalizer = ProsodyNormalizer.fit(
        [{"duration_s": 1.0, "f0_valid": True}, {"duration_s": 3.0, "f0_valid": False}],
        split_name="train",
        training_record_ids=["b", "a"],
    )
    path = normalizer.save(tmp_path / "prosody.json")
    restored = ProsodyNormalizer.load(path)

    assert restored.transform({"duration_s": 2.0, "f0_valid": True}) == {
        "duration_s": 0.0,
        "f0_valid": True,
    }
    assert restored.training_record_ids == ("a", "b")
    with pytest.raises(ValueError, match="training split"):
        ProsodyNormalizer.fit([], split_name="test", training_record_ids=["a"])


def test_optional_spectral_hook_can_be_absent() -> None:
    assert extract_optional_spectral_features(None, __import__("numpy").zeros(10), 16_000, []) == []
