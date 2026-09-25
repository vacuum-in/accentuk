"""Deterministic leakage-safe dataset split utilities."""

from ukstress.splits.core import (
    SplitReport,
    balance_by_lexeme,
    deterministic_split,
    speaker_held_out_split,
    target_word_held_out_split,
    validate_group_leakage,
)

__all__ = [
    "SplitReport",
    "balance_by_lexeme",
    "deterministic_split",
    "speaker_held_out_split",
    "target_word_held_out_split",
    "validate_group_leakage",
]
