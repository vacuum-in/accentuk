"""Backend-neutral word and fine alignment."""

from ukstress.alignment.canary_ctc_backend import CanaryCTCWordAligner
from ukstress.alignment.fine_interface import (
    CTCFineAlignerPlaceholder,
    FineAlignmentBackend,
    FineAlignmentResult,
)
from ukstress.alignment.fine_validation import (
    VowelAlignmentValidation,
    fine_alignment_rejection,
    validate_vowel_intervals,
)
from ukstress.alignment.interface import WordAlignmentBackend, WordAlignmentResult
from ukstress.alignment.mfa_backend import MFAFineAligner
from ukstress.alignment.stress_ctc import StressCTCAdapter, StressCTCOutput, StressCTCVote
from ukstress.alignment.whisperx_backend import WhisperXWordAligner
from ukstress.alignment.whisperx_char_backend import WhisperXCharFineAligner

__all__ = [
    "CTCFineAlignerPlaceholder",
    "CanaryCTCWordAligner",
    "FineAlignmentBackend",
    "FineAlignmentResult",
    "MFAFineAligner",
    "StressCTCAdapter",
    "StressCTCOutput",
    "StressCTCVote",
    "VowelAlignmentValidation",
    "WhisperXCharFineAligner",
    "WhisperXWordAligner",
    "WordAlignmentBackend",
    "WordAlignmentResult",
    "fine_alignment_rejection",
    "validate_vowel_intervals",
]
