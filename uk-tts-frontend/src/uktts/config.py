"""Environment-driven configuration for the unified TTS text frontend."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# There is no sensible default for the checkpoint: it is a multi-hundred-MB
# artifact whose location is a property of the deployment, not of the code.
# Leaving it unset fails loudly at startup rather than quietly serving
# unverbalized text.
DEFAULT_STRESS_URL = "http://127.0.0.1:8080"


def _flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return default if raw is None or not raw.strip() else int(raw)


def _float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return default if raw is None or not raw.strip() else float(raw)


@dataclass(frozen=True)
class VerbalizerConfig:
    """Where the Marian verbalizer lives and how it is decoded.

    `max_source_tokens` is the checkpoint's training window. Text longer than
    that is *chunked* before inference, never truncated: a silently truncated
    sentence loses words that the listener then never hears.
    """

    checkpoint: Path | None = None
    device: str = "auto"
    batch_size: int = 8
    num_beams: int = 1
    max_source_tokens: int = 384
    max_new_tokens: int = 384
    enabled: bool = True
    # Send a chunk to the model only when it has something to spell out.
    # Off, the checkpoint rewrites roughly a quarter of plain Ukrainian
    # sentences into something else; see route.py for the measurement.
    route: bool = True

    @classmethod
    def from_env(cls) -> VerbalizerConfig:
        raw = os.environ.get("UKTTS_VERBALIZER_CHECKPOINT")
        return cls(
            checkpoint=Path(raw) if raw else None,
            device=os.environ.get("UKTTS_VERBALIZER_DEVICE", "auto"),
            batch_size=_int("UKTTS_BATCH_SIZE", 8),
            num_beams=_int("UKTTS_NUM_BEAMS", 1),
            max_source_tokens=_int("UKTTS_MAX_SOURCE_TOKENS", 384),
            max_new_tokens=_int("UKTTS_MAX_NEW_TOKENS", 384),
            enabled=_flag("UKTTS_VERBALIZE", True),
            route=_flag("UKTTS_ROUTE", True),
        )


@dataclass(frozen=True)
class StressConfig:
    """How to reach the Go stress API.

    `on_ambiguity` is passed straight through: `default` serves the lexicon's
    highest-confidence reading for a word nothing decided, `preserve` emits it
    unstressed. TTS wants a decision, so `default` is the default here.
    """

    base_url: str = DEFAULT_STRESS_URL
    on_ambiguity: str = "default"
    timeout: float = 30.0
    enabled: bool = True

    @classmethod
    def from_env(cls) -> StressConfig:
        return cls(
            base_url=os.environ.get("UKTTS_STRESS_URL", DEFAULT_STRESS_URL).rstrip("/"),
            on_ambiguity=os.environ.get("UKTTS_ON_AMBIGUITY", "default"),
            timeout=_float("UKTTS_STRESS_TIMEOUT", 30.0),
            enabled=_flag("UKTTS_STRESS", True),
        )


@dataclass(frozen=True)
class Config:
    verbalizer: VerbalizerConfig = field(default_factory=VerbalizerConfig)
    stress: StressConfig = field(default_factory=StressConfig)

    @classmethod
    def from_env(cls) -> Config:
        return cls(verbalizer=VerbalizerConfig.from_env(), stress=StressConfig.from_env())
