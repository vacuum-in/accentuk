from pathlib import Path

import pytest

from ukstress.datasets import StressCandidate
from ukstress.manual_gold import (
    ManualGoldOccurrence,
    import_manual_gold,
    manual_gold_report,
    validate_manual_gold,
)


class Lexicon:
    def lookup(self, _word: str):
        return (
            StressCandidate(stressed_form="за́мок", vowel_index=0, source="fixture"),
            StressCandidate(stressed_form="замо́к", vowel_index=1, source="fixture"),
        )


def _record(record_id: str = "r") -> ManualGoldOccurrence:
    return ManualGoldOccurrence(
        record_id=record_id,
        source_id="s",
        utterance_id="u",
        target_word="замок",
        selected_vowel_index=1,
        context_original="Замок",
        context_normalized="замок",
        speaker_id="speaker",
        annotator_id="a",
        annotation_version="1",
    )


def test_manual_gold_validation_and_report() -> None:
    record = _record()
    validate_manual_gold([record], Lexicon())
    assert manual_gold_report([record])["forms"] == {"замок": {1: 1}}
    with pytest.raises(ValueError, match="duplicate"):
        validate_manual_gold([record, record], Lexicon())


def test_manual_gold_import_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "gold.jsonl"
    path.write_text(_record().model_dump_json() + "\n", encoding="utf-8")
    assert import_manual_gold(path)[0].record_id == "r"
