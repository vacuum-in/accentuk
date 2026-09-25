"""Backend-neutral automatic speech recognition."""

from ukstress.asr.interface import ASRBackend, ASRResult, ASRSegment, ASRToken
from ukstress.asr.whisper_backend import WhisperCompatibleBackend

__all__ = [
    "ASRBackend",
    "ASRResult",
    "ASRSegment",
    "ASRToken",
    "WhisperCompatibleBackend",
]
