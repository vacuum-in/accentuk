"""Normalized ASR records and backend protocol."""

from __future__ import annotations

from typing import Protocol, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ASRModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class ASRToken(ASRModel):
    text: str
    start_s: float | None = Field(default=None, ge=0.0)
    end_s: float | None = Field(default=None, gt=0.0)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def times_are_ordered(self) -> Self:
        if (self.start_s is None) != (self.end_s is None):
            raise ValueError("token start and end must be specified together")
        if self.start_s is not None and self.end_s is not None and self.start_s >= self.end_s:
            raise ValueError("token start must be before end")
        return self


class ASRSegment(ASRModel):
    text: str
    start_s: float = Field(ge=0.0)
    end_s: float = Field(gt=0.0)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    tokens: list[ASRToken] = Field(default_factory=list)

    @model_validator(mode="after")
    def times_are_ordered(self) -> Self:
        if self.start_s >= self.end_s:
            raise ValueError("segment start must be before end")
        return self


class ASRResult(ASRModel):
    transcript: str
    language: str
    backend: str
    model_id: str
    segments: list[ASRSegment]


class ASRBackend(Protocol):
    @property
    def backend_id(self) -> str: ...

    @property
    def model_id(self) -> str: ...

    def transcribe(self, waveform: np.ndarray, sample_rate: int) -> ASRResult: ...
