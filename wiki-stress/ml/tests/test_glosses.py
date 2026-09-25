import json
import unicodedata
from pathlib import Path

import pytest

from ukstress_ml.glosses import (
    AuthoredGloss,
    GlossError,
    GlossTask,
    apply_glosses,
    build_prompt,
    parse_response,
    validate_gloss,
)

TASK = GlossTask(
    group_id=7,
    missing=({"sense_id": "7.b", "stressed_example": "за́мок", "pos": "noun"},),
    known_siblings=({"sense_id": "7.a", "definition": "пристрій для замикання дверей"},),
)


def test_valid_gloss_is_normalised() -> None:
    assert validate_gloss("  укріплена   споруда феодальної доби  ") == (
        "укріплена споруда феодальної доби"
    )


def test_gloss_naming_the_stress_is_refused() -> None:
    """A gloss describing pronunciation leaks the label into the model input."""
    with pytest.raises(GlossError, match="describes stress"):
        validate_gloss("значення з наголосом на першому складі")


def test_gloss_quoting_a_stressed_word_keeps_the_definition_but_drops_the_mark() -> None:
    """Quoting a word with its acute is not a metalinguistic gloss, but leaving
    the mark in would hand the model the answer it must infer from context."""
    cleaned = validate_gloss("укріплена споруда, за́мок феодала")
    assert cleaned == "укріплена споруда, замок феодала"
    assert "́" not in unicodedata.normalize("NFD", cleaned)


def test_empty_and_trivial_glosses_are_refused() -> None:
    with pytest.raises(GlossError, match="empty"):
        validate_gloss("   ")
    with pytest.raises(GlossError, match="too short"):
        validate_gloss("будівля")


def test_gloss_that_only_restates_the_word_is_refused() -> None:
    with pytest.raises(GlossError, match="restates the word"):
        validate_gloss("замок замок", surface_forms=["за́мок"])


def test_prompt_shows_siblings_so_the_gloss_is_contrastive() -> None:
    prompt = build_prompt(TASK)
    assert "пристрій для замикання дверей" in prompt
    assert "7.b" in prompt
    assert "НЕ згадуй наголос" in prompt


def test_parse_accepts_a_valid_item() -> None:
    accepted, rejected = parse_response(
        TASK, '{"items":[{"sense_id":"7.b","definition":"укріплена споруда феодальної доби"}]}'
    )
    assert rejected == []
    assert accepted == [
        AuthoredGloss(7, "7.b", "укріплена споруда феодальної доби", "llm_proposed")
    ]


def test_parse_marks_every_authored_gloss_for_review() -> None:
    """An LLM gloss is a label definition and must never be silent ground truth."""
    accepted, _ = parse_response(
        TASK, '{"items":[{"sense_id":"7.b","definition":"укріплена споруда феодальної доби"}]}'
    )
    assert accepted[0].review_status == "llm_proposed"


def test_parse_rejects_leaking_declined_and_unknown_senses() -> None:
    accepted, rejected = parse_response(
        TASK,
        json.dumps(
            {
                "items": [
                    {"sense_id": "7.b", "definition": "наголос падає на другий склад"},
                    {"sense_id": "9.z", "definition": "щось інше зовсім"},
                ]
            }
        ),
    )
    assert accepted == []
    assert any("describes stress" in r for r in rejected)
    assert any("not a requested sense" in r for r in rejected)


def test_parse_reports_a_declined_sense() -> None:
    accepted, rejected = parse_response(
        TASK, '{"items":[{"sense_id":"7.b","definition":"","unclear":true}]}'
    )
    assert accepted == []
    assert any("declined" in r for r in rejected)


def test_non_json_response_is_reported_not_crashed() -> None:
    accepted, rejected = parse_response(TASK, "вибачте, не можу")
    assert accepted == []
    assert rejected == ["response contained no JSON object"]


def test_apply_glosses_fills_only_the_empty_ones(tmp_path: Path) -> None:
    inventory = tmp_path / "inv.jsonl"
    inventory.write_text(
        json.dumps(
            {
                "group_id": 7,
                "form": "замок",
                "candidates": [
                    {"sense_id": "7.a", "definition": "пристрій для замикання"},
                    {"sense_id": "7.b", "definition": ""},
                ],
                "complete": False,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "filled.jsonl"
    counts = apply_glosses(
        inventory, [AuthoredGloss(7, "7.b", "укріплена споруда феодальної доби")], out
    )

    assert counts == {"filled": 1, "already_present": 1, "still_missing": 0}
    row = json.loads(out.read_text(encoding="utf-8"))
    assert row["candidates"][0]["definition"] == "пристрій для замикання"
    assert row["candidates"][1]["definition"] == "укріплена споруда феодальної доби"
    assert row["candidates"][1]["review_status"] == "llm_proposed"
    assert row["complete"] is True


def test_apply_glosses_reports_senses_that_stay_empty(tmp_path: Path) -> None:
    inventory = tmp_path / "inv.jsonl"
    inventory.write_text(
        json.dumps(
            {"group_id": 7, "form": "з", "candidates": [{"sense_id": "7.b", "definition": ""}]},
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "filled.jsonl"
    counts = apply_glosses(inventory, [], out)
    assert counts["still_missing"] == 1
    assert json.loads(out.read_text(encoding="utf-8"))["complete"] is False


def test_composition_sense_of_sklad_is_not_mistaken_for_syllable() -> None:
    """Ukrainian «склад» means both *syllable* and *composition*. Matching the
    bare word rejected ordinary definitions as if they described stress."""
    assert validate_gloss("Бути складовою частиною чого-небудь, належати до складу.")
    assert validate_gloss("приміщення для зберігання товарів, склад")


def test_ordinal_before_sklad_is_still_caught() -> None:
    for leak in (
        "прикметник, пов'язаний із селом Боблів (на першому складі)",
        "те саме, але з наголосом на другому складі",
    ):
        with pytest.raises(GlossError, match="describes stress"):
            validate_gloss(leak)
