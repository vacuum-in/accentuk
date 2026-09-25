"""Word alignment contract shared by WhisperX/CTC and future backends."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ukstress.datasets import WordAlignment


class WordAlignmentResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    backend: str = Field(min_length=1)
    model_id: str | None = None
    utterance_duration_s: float = Field(gt=0.0)
    words: list[WordAlignment]
    quality: float = Field(default=0.0, ge=0.0, le=1.0)


@runtime_checkable
class WordAlignmentBackend(Protocol):
    @property
    def backend_id(self) -> str: ...

    @property
    def model_id(self) -> str | None: ...

    def align(
        self,
        waveform: np.ndarray,
        sample_rate: int,
        normalized_transcript: str,
    ) -> WordAlignmentResult: ...
