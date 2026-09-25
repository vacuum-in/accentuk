"""Speech representation and pooling helpers."""

from ukstress.features.attention import LearnableAttentionPooler, attention_pool_intervals
from ukstress.features.batching import (
    DurationBatch,
    batch_for_device,
    batch_indices_by_duration,
    inference_autocast,
)
from ukstress.features.cache import CachedFeatureArtifact, FeatureCache, feature_cache_key
from ukstress.features.normalization import ProsodyNormalizer
from ukstress.features.prosody import extract_prosodic_features, f0_track
from ukstress.features.spectral import SpectralFeatureHook, extract_optional_spectral_features
from ukstress.features.ssl import (
    HuggingFaceSpeechEncoder,
    SpeechEncoder,
    mean_pool_intervals,
    time_to_frame_bounds,
)

__all__ = [
    "CachedFeatureArtifact",
    "DurationBatch",
    "FeatureCache",
    "HuggingFaceSpeechEncoder",
    "LearnableAttentionPooler",
    "ProsodyNormalizer",
    "SpectralFeatureHook",
    "SpeechEncoder",
    "attention_pool_intervals",
    "batch_for_device",
    "batch_indices_by_duration",
    "extract_optional_spectral_features",
    "extract_prosodic_features",
    "f0_track",
    "feature_cache_key",
    "inference_autocast",
    "mean_pool_intervals",
    "time_to_frame_bounds",
]
