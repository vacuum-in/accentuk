import unicodedata

import pytest
from hypothesis import given
from hypothesis import strategies as st

from ukstress.normalizer import (
    ACUTE,
    CANONICAL_APOSTROPHE,
    NormalizationError,
    canonical_stressed_form,
    lookup_key,
    stress_signature,
    validate_characters,
    validate_stress,
)


def test_nfd_and_acute_are_preserved_in_stressed_form() -> None:
    assert canonical_stressed_form("мо\u0301ва") == unicodedata.normalize("NFD", "мо\u0301ва")
    assert lookup_key("мо\u0301ва") == "мова"
    assert ACUTE not in lookup_key("мо\u0301ва")


def test_apostrophes_and_internal_hyphens_are_canonicalized() -> None:
    assert lookup_key("п'ять") == f"п{CANONICAL_APOSTROPHE}ять"
    assert lookup_key("пів—яблука") == "пів-яблука"


def test_lookup_key_does_not_remove_unrelated_combining_marks() -> None:
    assert lookup_key("а\u0308") == "а\u0308"


@pytest.mark.parametrize(
    ("value", "reason"),
    [("\u0301мова", "acute"), ("м\u0301ова", "acute"), ("мова\x00", "control")],
)
def test_invalid_stress_and_controls_are_rejected(value: str, reason: str) -> None:
    result = validate_stress(value)
    assert result.classification == "rejected"
    assert reason in (result.reason or "")


def test_stress_on_decomposing_vowels() -> None:
    # `ї` decomposes to `і` plus a combining diaeresis, so the code point before
    # the acute is a mark rather than the stressed letter.
    assert validate_stress("Украї́на").classification == "valid"
    assert validate_stress("ї́хати").classification == "valid"
    assert stress_signature("краї́на") == "1"
    # `й` decomposes the same way but is not a vowel and never carries stress.
    assert validate_stress("бай́ка").classification == "rejected"


def test_ukrainian_character_validation() -> None:
    assert validate_characters("ґанок").classification == "valid"
    assert validate_characters("кра́щий").classification == "valid"
    assert validate_characters("ї́хній").classification == "valid"
    assert validate_characters("word").classification == "rejected"


def test_stress_signatures_use_vowel_ordinals() -> None:
    assert stress_signature("мо\u0301ва") == "0"
    assert stress_signature("вода\u0301") == "1"
    assert stress_signature("пере́кладати") == "1"
    assert stress_signature("переклада́ти") == "3"
    assert stress_signature("до́бре-зна́ти") == "0:0|1:0"


def test_signature_requires_stress() -> None:
    with pytest.raises(NormalizationError, match="no acute"):
        stress_signature("мова")


@given(st.text(alphabet="абвгґдеєжзиіїйклмнопрстуфхцчшщьюя'`—- \u0301", min_size=0, max_size=40))
def test_canonicalization_is_idempotent(value: str) -> None:
    assert canonical_stressed_form(canonical_stressed_form(value)) == canonical_stressed_form(value)
