"""Backend-neutral contract for phone and vowel-level alignment."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ukstress.datasets import VowelInterval, WordAlignment


class FineAlignmentResult(BaseModel):
    """Canonical result returned by a fine phone/vowel aligner."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    backend: str = Field(min_length=1)
    model_id: str | None = None
    target_word: str = Field(min_length=1)
    word_start_s: float = Field(ge=0.0)
    word_end_s: float = Field(gt=0.0)
    vowels: list[VowelInterval]
    quality: float = Field(default=0.0, ge=0.0, le=1.0)
    rejection_reasons: list[str] = Field(default_factory=list)

    def with_rejections(self, reasons: list[str]) -> FineAlignmentResult:
        """Return an immutable result carrying machine-readable rejection reasons."""

        return self.model_copy(update={"rejection_reasons": list(dict.fromkeys(reasons))})


@runtime_checkable
class FineAlignmentBackend(Protocol):
    """Interface consumed by feature extraction and validation layers."""

    @property
    def backend_id(self) -> str: ...

    @property
    def model_id(self) -> str | None: ...

    def align(
        self,
        waveform: np.ndarray,
        sample_rate: int,
        word: WordAlignment,
        normalized_word: str,
    ) -> FineAlignmentResult: ...


class CTCFineAlignerPlaceholder:
    """Explicit placeholder for a future scalable phone/CTC fine aligner."""

    backend_id = "ctc_phone"
    model_id: str | None = None

    def align(
        self,
        waveform: np.ndarray,
        sample_rate: int,
        word: WordAlignment,
        normalized_word: str,
    ) -> FineAlignmentResult:
        del waveform, sample_rate, word, normalized_word
        raise RuntimeError(
            "the ctc_phone fine aligner is not implemented; configure MFA or provide a backend"
        )
