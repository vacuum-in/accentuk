import math

import numpy as np
import pytest

from ukstress.datasets import VowelInterval
from ukstress.features import (
    FeatureCache,
    LearnableAttentionPooler,
    attention_pool_intervals,
    batch_indices_by_duration,
    extract_prosodic_features,
    feature_cache_key,
)


def _vowels() -> list[VowelInterval]:
    return [
        VowelInterval(vowel_index=0, grapheme="а", start_s=0.10, end_s=0.20),
        VowelInterval(vowel_index=1, grapheme="о", start_s=0.30, end_s=0.50),
    ]


def test_attention_pooling_supports_context_and_has_trainable_parameters() -> None:
    pytest.importorskip("torch")
    pooler = LearnableAttentionPooler(3)
    embeddings = np.arange(30, dtype=np.float32).reshape(10, 3)
    pooled = attention_pool_intervals(
        embeddings, _vowels(), pooler, frame_shift_s=0.1, context_ms=50
    )

    assert len(pooled) == 2 and all(len(vector) == 3 for vector in pooled)
    assert sum(parameter.numel() for parameter in pooler.parameters()) > 0


def test_duration_batching_is_stable_and_bounded() -> None:
    batches = batch_indices_by_duration([3.0, 1.0, 2.0, 1.5], max_batch_duration_s=3.0)
    assert [batch.indices for batch in batches] == [(1, 3), (2,), (0,)]
    assert all(batch.total_duration_s <= 3.0 for batch in batches)


def test_feature_cache_is_keyed_and_round_trips_embeddings(tmp_path) -> None:
    key = feature_cache_key(
        encoder_id="encoder", config_fingerprint="sha256:c", input_fingerprint="sha256:i"
    )
    cache = FeatureCache(tmp_path)
    stored = cache.store(key, np.ones((2, 3), dtype=np.float32), metadata={"model": "encoder"})
    loaded = cache.load(key)

    assert stored == cache.path_for(key)
    assert loaded is not None and loaded.metadata == {"model": "encoder"}
    assert loaded.embeddings.shape == (2, 3)


def test_prosody_extracts_duration_energy_and_voiced_f0() -> None:
    sample_rate = 16_000
    seconds = np.arange(sample_rate, dtype=np.float32) / sample_rate
    waveform = (0.2 * np.sin(2 * np.pi * 200 * seconds)).astype(np.float32)
    features = extract_prosodic_features(
        waveform, sample_rate, _vowels(), word_start_s=0.10, word_end_s=0.50,
        utterance_vowels=_vowels(),
    )

    assert features[0]["duration_s"] == pytest.approx(0.1)
    assert features[1]["log_duration"] == pytest.approx(math.log(0.2))
    assert features[0]["duration_relative_word"] == pytest.approx(0.5)
    assert features[0]["rms_energy"] == pytest.approx(0.2 / math.sqrt(2), rel=0.05)
    assert features[0]["f0_valid"] is True
    assert features[0]["f0_median_hz"] == pytest.approx(200.0, rel=0.08)
