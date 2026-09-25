import pytest

from ukstress.datasets.ids import dataset_fingerprint, stable_record_id


def _record_id(**overrides: object) -> str:
    identity: dict[str, object] = {
        "source_id": "commonvoice",
        "utterance_id": "utt-1",
        "normalized_target": "замок",
        "word_start_s": 1.0001,
        "word_end_s": 1.5001,
    }
    identity.update(overrides)
    return stable_record_id(**identity)


def test_record_id_is_stable_within_millisecond_quantization() -> None:
    assert _record_id() == _record_id(word_start_s=1.0004, word_end_s=1.5004)
    assert _record_id().startswith("rec_")


def test_record_id_changes_across_identity_boundaries() -> None:
    assert _record_id() != _record_id(utterance_id="utt-2")
    assert _record_id() != _record_id(word_start_s=1.002)


def test_record_id_normalizes_unicode_and_case() -> None:
    assert _record_id(normalized_target="ЗАМОК") == _record_id(normalized_target="замок")


def test_record_id_rejects_interval_collapsed_by_quantization() -> None:
    with pytest.raises(ValueError, match="quantized word start"):
        _record_id(word_start_s=1.0001, word_end_s=1.0002)


def test_dataset_fingerprint_preserves_record_order_and_split_semantics() -> None:
    common = {
        "lexicon_fingerprint": "sha256:lexicon",
        "normalization_version": "uk-text-v1",
        "split_config": {"seed": 17},
    }
    first = dataset_fingerprint(ordered_record_ids=["a", "b"], **common)

    assert first == dataset_fingerprint(ordered_record_ids=["a", "b"], **common)
    assert first != dataset_fingerprint(ordered_record_ids=["b", "a"], **common)
    assert first != dataset_fingerprint(
        ordered_record_ids=["a", "b"],
        **{**common, "split_config": {"seed": 18}},
    )
