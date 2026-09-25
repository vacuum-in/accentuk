"""End-to-end checks against the real checkpoint. Skipped when it is absent."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from uktts.config import Config, StressConfig, VerbalizerConfig
from uktts.pipeline import Pipeline
from uktts.stress import PassthroughStresser
from uktts.verbalize import MarianVerbalizer, default_checkpoint_exists

CHECKPOINT = Path(os.environ["UKTTS_VERBALIZER_CHECKPOINT"]) if os.environ.get(
    "UKTTS_VERBALIZER_CHECKPOINT"
) else None

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        CHECKPOINT is None or not default_checkpoint_exists(CHECKPOINT),
        reason="set UKTTS_VERBALIZER_CHECKPOINT to a Marian checkpoint to run these",
    ),
]


@pytest.fixture(scope="module")
def pipeline() -> Pipeline:
    config = Config(
        verbalizer=VerbalizerConfig(checkpoint=CHECKPOINT, device="cpu"),
        stress=StressConfig(enabled=False),
    )
    return Pipeline(config, stresser=PassthroughStresser())


def test_a_missing_checkpoint_fails_at_construction() -> None:
    with pytest.raises(FileNotFoundError):
        MarianVerbalizer(VerbalizerConfig(checkpoint=CHECKPOINT / "nowhere"))


def test_digits_are_read_as_words(pipeline: Pipeline) -> None:
    result = pipeline.prepare("Ціна 1 500 грн.")
    assert not any(character.isdigit() for character in result.text)
    assert "гривень" in result.text


def test_time_is_read_as_words(pipeline: Pipeline) -> None:
    result = pipeline.prepare("Зустріч о 14:30.")
    assert ":" not in result.text
    assert "чотирнадцятій" in result.text


def test_latin_is_transliterated(pipeline: Pipeline) -> None:
    result = pipeline.prepare("Компанія Apple випустила iPhone 15 Pro.")
    assert not any("a" <= character.lower() <= "z" for character in result.text)


def test_paragraph_structure_is_preserved(pipeline: Pipeline) -> None:
    result = pipeline.prepare("Перше речення.\n\nДруге речення.")
    assert result.verbalized.count("\n\n") == 1


def test_text_longer_than_the_source_window_is_not_truncated(pipeline: Pipeline) -> None:
    text = " ".join(f"Це коротке речення номер {index}." for index in range(1, 61))
    result = pipeline.prepare(text)
    assert result.chunks == 60
    assert len(result.text.split()) > 240


def test_an_unconfigured_checkpoint_names_the_setting() -> None:
    with pytest.raises(FileNotFoundError, match="UKTTS_VERBALIZER_CHECKPOINT"):
        MarianVerbalizer(VerbalizerConfig(checkpoint=None))
