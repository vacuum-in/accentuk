"""Configuration models and deterministic YAML loading."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


class StrictModel(BaseModel):
    """Base model that makes configuration mistakes fail early."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class AlignmentConfig(StrictModel):
    word_backend: str = "whisperx"
    word_model_id: str = "Yehor/wav2vec2-xls-r-300m-uk-with-small-lm"
    language: str = "uk"
    device: str = "cpu"
    model_cache_dir: Path | None = None
    model_cache_only: bool = False
    return_char_alignments: bool = True
    model_filename: str = "canary-ctc-aligner-q8_0.gguf"
    model_revision: str = "2b22dc9aff585bc8368a5228173c8afe22c155b7"
    model_path: Path | None = None
    n_threads: int = Field(default=4, ge=1)
    fine_backend: str = "whisperx_chars"
    min_quality: float = Field(default=0.8, ge=0.0, le=1.0)
    overlap_tolerance_s: float = Field(default=0.005, ge=0.0)


class ProsodyConfig(StrictModel):
    duration: bool = True
    energy: bool = True
    f0: bool = True
    f0_min_hz: float = Field(default=60.0, gt=0.0)
    f0_max_hz: float = Field(default=400.0, gt=0.0)
    f0_frame_ms: int = Field(default=40, ge=10, le=200)
    f0_hop_ms: int = Field(default=10, ge=1, le=100)

    @model_validator(mode="after")
    def f0_range_is_ordered(self) -> ProsodyConfig:
        if self.f0_min_hz >= self.f0_max_hz:
            raise ValueError("prosody.f0_min_hz must be below prosody.f0_max_hz")
        return self


class FeaturesConfig(StrictModel):
    ssl_model: str | None = None
    vowel_context_ms: int = Field(default=40, ge=0, le=500)
    device: str = "cpu"
    mixed_precision: bool = False
    max_batch_duration_s: float = Field(default=60.0, gt=0.0)
    cache_dir: Path | None = None
    prosody: ProsodyConfig = Field(default_factory=ProsodyConfig)


class ModelConfig(StrictModel):
    hidden_size: int = Field(default=256, gt=0)
    transformer_layers: int = Field(default=2, ge=1)
    attention_heads: int = Field(default=4, ge=1)
    ffn_size: int = Field(default=768, gt=0)
    trainable_ssl_layers: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def hidden_size_must_fit_heads(self) -> ModelConfig:
        if self.hidden_size % self.attention_heads != 0:
            raise ValueError("model.hidden_size must be divisible by model.attention_heads")
        return self


class TrainingConfig(StrictModel):
    batch_size: int = Field(default=16, ge=1)
    learning_rate: float = Field(default=3e-4, gt=0.0)
    weight_decay: float = Field(default=1e-2, ge=0.0)
    max_grad_norm: float = Field(default=1.0, gt=0.0)
    scheduler: str = "cosine"
    mixed_precision: bool = False


class MiningConfig(StrictModel):
    require_candidate_scorer_agreement: bool = False
    stress_ctc_disagreement: Literal["reject", "penalize", "ignore"] = "reject"
    stress_ctc_penalty: float = Field(default=0.0, ge=0.0, le=1.0)
    min_calibrated_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    min_candidate_margin: float | None = Field(default=None, ge=0.0, le=1.0)
    min_identity_quality: float = Field(default=1.0, ge=0.0, le=1.0)


class LexiconConfig(StrictModel):
    path: Path
    version: str = Field(min_length=1)
    source_name: str | None = None


class ASRConfig(StrictModel):
    backend: str = "faster-whisper"
    model_id: str = "large-v3"
    language: str = "uk"
    device: str = "auto"
    compute_type: str = "default"
    beam_size: int = Field(default=5, ge=1)


class AppConfig(StrictModel):
    schema_version: str = "1.0"
    sample_rate: int = Field(default=16_000, ge=8_000, le=192_000)
    seed: int = Field(default=17, ge=0)
    lexicon: LexiconConfig | None = None
    asr: ASRConfig = Field(default_factory=ASRConfig)
    alignment: AlignmentConfig = Field(default_factory=AlignmentConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    mining: MiningConfig = Field(default_factory=MiningConfig)


def _expand_environment(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand_environment(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_environment(item) for item in value]
    if not isinstance(value, str):
        return value

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        try:
            return os.environ[name]
        except KeyError as error:
            message = f"configuration references unset environment variable {name}"
            raise ValueError(message) from error

    return _ENV_PATTERN.sub(replace, value)


def load_config(path: str | Path) -> AppConfig:
    """Load a YAML mapping, expand ``${VAR}``, and validate it strictly."""

    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("configuration root must be a YAML mapping")
    return AppConfig.model_validate(_expand_environment(raw))
