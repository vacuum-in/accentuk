import json
from pathlib import Path

from ukstress_ml.paradigm import (
    UNMAPPED_COUNT_MISMATCH,
    UNMAPPED_NO_GROUP,
    ExpandedForm,
    UnmappedForm,
    expand_form,
    iter_expansions,
    signature_sort_key,
    write_expansion,
)

SENSES = [
    {"sense_id": "1.a", "signature": "1", "definition": "стем-наголос", "pos": "adj", "priority": 0},
    {"sense_id": "1.b", "signature": "2", "definition": "закінчення", "pos": "adj", "priority": 3},
]


def test_single_token_signatures_sort_by_vowel_position() -> None:
    assert signature_sort_key("0") < signature_sort_key("1") < signature_sort_key("10")


def test_compound_signatures_sort_by_segment_then_position() -> None:
    """The bug the ground-truth check caught: compound signatures all collapsed
    to one key, so senses paired arbitrarily."""
    assert signature_sort_key("0:0|1:1") != signature_sort_key("1:1")
    assert signature_sort_key("0:0|1:1") < signature_sort_key("0:0|1:2")
    assert signature_sort_key("0:1|1:0") > signature_sort_key("0:0|1:0")


def test_senses_pair_with_signatures_in_stress_position_order() -> None:
    result = expand_form(
        "відрубною", "відрубний", 7, SENSES,
        {"2": "відрубно́ю", "1": "відру́бною"}, paradigm_source="lemma",
    )
    assert isinstance(result, ExpandedForm)
    assert [(c.sense_id, c.signature, c.stressed) for c in result.candidates] == [
        ("1.a", "1", "відру́бною"),
        ("1.b", "2", "відрубно́ю"),
    ]
    # Glosses are inherited from the lemma's senses, which is the whole point.
    assert result.candidates[0].definition == "стем-наголос"
    assert result.group_id == 7


def test_input_order_does_not_affect_the_assignment() -> None:
    forward = expand_form("ф", "л", 1, SENSES, {"1": "a", "2": "b"}, paradigm_source="lemma")
    reversed_senses = expand_form(
        "ф", "л", 1, list(reversed(SENSES)), {"2": "b", "1": "a"}, paradigm_source="lemma"
    )
    assert isinstance(forward, ExpandedForm) and isinstance(reversed_senses, ExpandedForm)
    assert forward.candidates == reversed_senses.candidates


def test_count_mismatch_is_refused_rather_than_guessed() -> None:
    """Pairing 2 senses onto 3 signatures would silently mislabel a paradigm."""
    result = expand_form(
        "ф", "л", 1, SENSES, {"1": "a", "2": "b", "3": "c"}, paradigm_source="lemma"
    )
    assert isinstance(result, UnmappedForm)
    assert result.reason == UNMAPPED_COUNT_MISMATCH
    assert (result.sense_count, result.signature_count) == (2, 3)


def test_forms_without_a_group_are_reported_not_dropped() -> None:
    rows = list(iter_expansions([{"form": "невідоме", "lemma": "невідомий"}], {}, {}))
    assert len(rows) == 1
    assert isinstance(rows[0], UnmappedForm)
    assert rows[0].reason == UNMAPPED_NO_GROUP


def test_expansion_uses_the_lemma_group_for_an_inflected_form() -> None:
    inventory = {"відрубний": {"group_id": 7, "candidates": SENSES}}
    rows = list(
        iter_expansions(
            [{"form": "відрубною", "lemma": "відрубний"}],
            inventory,
            {"відрубною": {"1": "відру́бною", "2": "відрубно́ю"}},
        )
    )
    assert isinstance(rows[0], ExpandedForm)
    assert rows[0].paradigm_source == "lemma"
    assert rows[0].group_id == 7


def test_write_expansion_separates_accepted_from_refused(tmp_path: Path) -> None:
    path = tmp_path / "expanded.jsonl"
    expanded = [
        expand_form("ф", "л", 1, SENSES, {"1": "a", "2": "b"}, paradigm_source="lemma")
    ]
    unmapped = [UnmappedForm("г", "л", UNMAPPED_COUNT_MISMATCH, 2, 3)]
    counts = write_expansion(path, expanded, unmapped)  # type: ignore[arg-type]

    assert counts == {"expanded": 1, "candidates": 2, "unmapped": 1}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["form"] == "ф"
    refused = (tmp_path / "expanded_unmapped.jsonl").read_text(encoding="utf-8")
    assert UNMAPPED_COUNT_MISMATCH in refused
