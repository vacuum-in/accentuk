import numpy as np
import pytest

from ukstress.alignment import (
    CTCFineAlignerPlaceholder,
    FineAlignmentResult,
    fine_alignment_rejection,
    validate_vowel_intervals,
)
from ukstress.datasets import VowelInterval, WordAlignment


def _vowel(grapheme: str, start: float, end: float, index: int) -> VowelInterval:
    return VowelInterval(vowel_index=index, grapheme=grapheme, start_s=start, end_s=end)


def test_vowel_validation_accepts_ordered_intervals_inside_word() -> None:
    result = validate_vowel_intervals(
        [_vowel("а", 0.20, 0.30, 0), _vowel("о", 0.35, 0.45, 1)],
        "замок",
        word_start_s=0.15,
        word_end_s=0.50,
        utterance_duration_s=1.0,
    )

    assert result.valid
    assert result.expected_count == result.observed_count == 2
    assert result.reasons == []


def test_vowel_validation_reports_count_overlap_and_containment_errors() -> None:
    result = validate_vowel_intervals(
        [_vowel("а", 0.10, 0.30, 0)],
        "замок",
        word_start_s=0.20,
        word_end_s=0.40,
        utterance_duration_s=0.35,
    )

    assert not result.valid
    assert set(result.reasons) == {
        "vowel_count_mismatch",
        "vowel_outside_word",
        "vowel_overlap",
        "word_interval_out_of_bounds",
    }


def test_fine_rejection_is_persistable() -> None:
    validation = validate_vowel_intervals(
        [_vowel("а", 0.1, 0.2, 0)], "замок", word_start_s=0.1, word_end_s=0.3
    )
    rejection = fine_alignment_rejection(
        source_id="source", utterance_id="utt", record_id="record", validation=validation
    )
    assert rejection is not None and rejection.stage == "fine_alignment"


def test_fine_result_carries_rejection_reasons_and_ctc_placeholder_is_explicit() -> None:
    result = FineAlignmentResult(
        backend="fixture",
        target_word="замок",
        word_start_s=0.1,
        word_end_s=0.6,
        vowels=[_vowel("а", 0.2, 0.3, 0), _vowel("о", 0.4, 0.5, 1)],
    ).with_rejections(["vowel_count_mismatch", "vowel_count_mismatch"])

    assert result.rejection_reasons == ["vowel_count_mismatch"]
    word = WordAlignment(
        token="замок",
        normalized_token="замок",
        start_s=0.1,
        end_s=0.6,
        backend="fixture",
    )
    with pytest.raises(RuntimeError, match="ctc_phone"):
        CTCFineAlignerPlaceholder().align(np.zeros(16_000, dtype=np.float32), 16_000, word, "замок")
