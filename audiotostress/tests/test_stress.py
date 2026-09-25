import pytest

from ukstress.lexicon.stress import (
    apply_stress,
    canonicalize_stressed_form,
    normalize_surface,
    parse_stress,
    vowel_count,
)


def test_parse_stress_returns_zero_based_vowel_index() -> None:
    parsed = parse_stress("замо́к")

    assert parsed.surface == "замок"
    assert parsed.vowel_index == 1
    assert parsed.stressed_form == "замо́к"


def test_common_stress_notations_are_canonicalized() -> None:
    assert canonicalize_stressed_form("ЗАМ+ОК") == "замо́к"
    assert canonicalize_stressed_form("замо´к") == "замо́к"


def test_surface_normalizes_apostrophe_hyphen_case_and_unicode() -> None:
    assert normalize_surface("ОБ’ЄКТ") == "об'єкт"
    assert normalize_surface("БУДЬ—ЯКИЙ") == "будь-який"
    assert normalize_surface("і\N{COMBINING DIAERESIS}") == "ї"


def test_apply_stress_and_vowel_count() -> None:
    assert vowel_count("будь-який") == 3
    assert apply_stress("будь-який", 1) == "будь-я́кий"


@pytest.mark.parametrize("form", ["замок", "за́мо́к", "+замок", "замок+"])
def test_invalid_stress_notation_is_rejected(form: str) -> None:
    with pytest.raises(ValueError):
        parse_stress(form)


def test_out_of_range_vowel_index_is_rejected() -> None:
    with pytest.raises(ValueError, match="outside"):
        apply_stress("замок", 2)
