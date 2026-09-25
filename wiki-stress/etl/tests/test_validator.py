import pytest

from ukstress.validator import classify_candidate


@pytest.mark.parametrize("value", ["́мова", "м́ова", "мо́ва<script>", "{{мо́ва}}", "сло́во\x00"])
def test_malformed_candidates_are_rejected(value: str) -> None:
    assert classify_candidate(value).classification == "rejected"


def test_unstressed_ukrainian_form_is_plausible() -> None:
    result = classify_candidate("мова")
    assert result.classification == "plausible"
    assert result.reasons == ("no explicit stress mark",)


def test_multiword_expression_is_valid_and_classified_separately() -> None:
    result = classify_candidate("офіці́йна мо́ва")
    assert result.classification == "valid"
    assert result.is_multiword is True

