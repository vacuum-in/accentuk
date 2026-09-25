from __future__ import annotations

import pytest

from uktts.config import Config, StressConfig, VerbalizerConfig
from uktts.pipeline import Pipeline
from uktts.stress import PassthroughStresser, StressResult, StressToken, StressUnavailable


class FakeVerbalizer:
    """Uppercases each chunk so a substitution is visible in the output."""

    name = "fake"

    def __init__(self, budget: int = 1000) -> None:
        self.calls: list[list[str]] = []
        self.budget = budget

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def verbalize(self, texts: list[str]) -> list[str]:
        self.calls.append(list(texts))
        return [text.upper() for text in texts]


class RecordingStresser:
    name = "recording"

    def __init__(self) -> None:
        self.seen: list[tuple[str, str | None]] = []

    def stress(self, text: str, on_ambiguity: str | None = None) -> StressResult:
        self.seen.append((text, on_ambiguity))
        return StressResult(
            text=text.replace("А", "А́"),
            tokens=(StressToken(0, 1, "а", "а́", "stressed"),),
            warnings=("model unavailable",),
        )

    def ready(self) -> tuple[bool, str]:
        return True, "ok"


def build(verbalizer=None, stresser=None, max_source_tokens: int = 1000,
          route: bool = False) -> Pipeline:
    # Routing is off here so the fake verbalizer sees every chunk; the
    # router has its own tests, and mixing the two would make a chunking
    # assertion depend on whether its text happens to contain a digit.
    config = Config(
        verbalizer=VerbalizerConfig(max_source_tokens=max_source_tokens, route=route),
        stress=StressConfig(),
    )
    return Pipeline(
        config,
        verbalizer=verbalizer or FakeVerbalizer(),
        stresser=stresser or PassthroughStresser(),
    )


def test_whitespace_and_punctuation_survive_the_round_trip() -> None:
    pipeline = build()
    result = pipeline.prepare("  Одне речення. Друге речення.  ")
    assert result.verbalized == "  ОДНЕ РЕЧЕННЯ. ДРУГЕ РЕЧЕННЯ.  "


def test_only_speakable_chunks_reach_the_model() -> None:
    verbalizer = FakeVerbalizer()
    build(verbalizer).prepare("Перше.\n\nДруге.")
    assert verbalizer.calls == [["Перше.", "Друге."]]


def test_several_inputs_share_one_model_pass() -> None:
    verbalizer = FakeVerbalizer()
    results = build(verbalizer).prepare_many(["Перше речення.", "Друге. Третє."])
    assert verbalizer.calls == [["Перше речення.", "Друге.", "Третє."]]
    assert [result.verbalized for result in results] == [
        "ПЕРШЕ РЕЧЕННЯ.",
        "ДРУГЕ. ТРЕТЄ.",
    ]
    assert [result.chunks for result in results] == [1, 2]


def test_stress_runs_on_verbalized_text_not_on_the_source() -> None:
    stresser = RecordingStresser()
    result = build(stresser=stresser).prepare("Абрикос.")
    assert stresser.seen == [("АБРИКОС.", None)]
    assert result.text == "А́БРИКОС."
    assert result.source == "Абрикос."


def test_on_ambiguity_is_passed_through_per_request() -> None:
    stresser = RecordingStresser()
    build(stresser=stresser).prepare("Слово.", on_ambiguity="preserve")
    assert stresser.seen == [("СЛОВО.", "preserve")]


def test_stress_warnings_reach_the_caller() -> None:
    result = build(stresser=RecordingStresser()).prepare("Слово.")
    assert "model unavailable" in result.warnings


def test_oversized_sentence_is_chunked_and_warned_about() -> None:
    verbalizer = FakeVerbalizer()
    pipeline = build(verbalizer, max_source_tokens=3)
    result = pipeline.prepare("Раз, два, три, чотири, п'ять.")
    assert result.chunks > 1
    assert any("source window" in warning for warning in result.warnings)
    assert result.verbalized == "РАЗ, ДВА, ТРИ, ЧОТИРИ, П'ЯТЬ."


def test_empty_input_makes_no_model_call() -> None:
    verbalizer = FakeVerbalizer()
    result = build(verbalizer).prepare("")
    assert verbalizer.calls == [[]]
    assert result.text == ""


class BrokenVerbalizer(FakeVerbalizer):
    def verbalize(self, texts: list[str]) -> list[str]:
        return []


def test_a_backend_returning_the_wrong_count_fails_loudly() -> None:
    with pytest.raises(RuntimeError, match="outputs for"):
        build(BrokenVerbalizer()).prepare("Речення.")


class DownStresser(PassthroughStresser):
    def stress(self, text: str, on_ambiguity: str | None = None) -> StressResult:
        raise StressUnavailable("stress API unreachable")


def test_stress_outage_propagates_rather_than_returning_unstressed_text() -> None:
    with pytest.raises(StressUnavailable):
        build(stresser=DownStresser()).prepare("Речення.")


def test_ready_reports_both_stages() -> None:
    status = build(stresser=RecordingStresser()).ready()
    assert status["ready"] is True
    assert status["verbalizer"] == "fake"
    assert status["stress"]["backend"] == "recording"


class PerCallStresser(RecordingStresser):
    """Marks the first vowel, so each call's output is visibly its own."""

    def stress(self, text: str, on_ambiguity: str | None = None) -> StressResult:
        self.seen.append((text, on_ambiguity))
        return StressResult(
            text=text.replace("Е", "Е́", 1),
            tokens=(StressToken(0, 4, text.split()[0], text.split()[0], "stressed"),),
        )


def test_each_sentence_is_stressed_in_its_own_call() -> None:
    stresser = PerCallStresser()
    build(stresser=stresser).prepare("Перше речення. Друге речення.")
    assert [call[0] for call in stresser.seen] == ["ПЕРШЕ РЕЧЕННЯ.", "ДРУГЕ РЕЧЕННЯ."]


def test_whitespace_between_sentences_is_not_sent_to_the_stress_service() -> None:
    stresser = PerCallStresser()
    build(stresser=stresser).prepare("Перше.\n\nДруге.")
    assert [call[0] for call in stresser.seen] == ["ПЕРШЕ.", "ДРУГЕ."]


def test_per_sentence_output_is_reassembled_with_its_separators() -> None:
    result = build(stresser=PerCallStresser()).prepare("Перше.\n\nДруге.")
    # Each call marked its own first vowel; the blank line between them is
    # carried through untouched.
    assert result.text == "ПЕ\u0301РШЕ.\n\nДРУГЕ\u0301."


def test_token_offsets_are_rebased_onto_the_whole_text() -> None:
    result = build(stresser=PerCallStresser()).prepare("Перше. Друге.")
    assert [(token.start, token.end) for token in result.tokens] == [(0, 4), (7, 11)]


def test_a_single_sentence_still_takes_one_call() -> None:
    stresser = PerCallStresser()
    build(stresser=stresser).prepare("Одне речення.")
    assert len(stresser.seen) == 1


def test_mode_both_runs_both_stages() -> None:
    verbalizer, stresser = FakeVerbalizer(), RecordingStresser()
    result = build(verbalizer, stresser).prepare("Абрикос.", mode="both")
    assert verbalizer.calls == [["Абрикос."]]
    assert stresser.seen == [("АБРИКОС.", None)]
    assert result.text == "А́БРИКОС."


def test_mode_verbalize_skips_the_stress_service() -> None:
    verbalizer, stresser = FakeVerbalizer(), RecordingStresser()
    result = build(verbalizer, stresser).prepare("Абрикос.", mode="verbalize")
    assert verbalizer.calls == [["Абрикос."]]
    assert stresser.seen == []
    assert result.verbalized == "АБРИКОС."
    assert result.text == "АБРИКОС."
    assert result.tokens == ()


def test_mode_stress_skips_the_verbalizer() -> None:
    verbalizer, stresser = FakeVerbalizer(), RecordingStresser()
    result = build(verbalizer, stresser).prepare("Абрикос.", mode="stress")
    assert verbalizer.calls == []
    assert stresser.seen == [("Абрикос.", None)]
    assert result.verbalized == "Абрикос."
    assert result.text == "А́брикос."


def test_stress_only_does_not_recut_a_long_sentence() -> None:
    # The 384-token window belongs to the verbalizer; with that stage off,
    # cutting at clause seams would only fragment the stress model's context.
    stresser = RecordingStresser()
    pipeline = build(FakeVerbalizer(), stresser, max_source_tokens=3)
    pipeline.prepare("Раз, два, три, чотири, п'ять.", mode="stress")
    assert stresser.seen == [("Раз, два, три, чотири, п'ять.", None)]


def test_an_unknown_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="mode must be one of"):
        build().prepare("Слово.", mode="phonemize")


def test_a_plain_sentence_is_not_sent_to_the_verbalizer() -> None:
    verbalizer = FakeVerbalizer()
    result = build(verbalizer, route=True).prepare("Він не має руки.")
    assert verbalizer.calls == [[]]
    assert result.verbalized == "Він не має руки."


def test_a_sentence_with_a_number_is_sent() -> None:
    verbalizer = FakeVerbalizer()
    build(verbalizer, route=True).prepare("Ціна 1 500 грн.")
    assert verbalizer.calls == [["Ціна 1 500 грн."]]


def test_only_the_chunks_that_need_it_are_sent() -> None:
    verbalizer = FakeVerbalizer()
    result = build(verbalizer, route=True).prepare("Просте речення. Ціна 500 грн.")
    assert verbalizer.calls == [["Ціна 500 грн."]]
    assert result.verbalized == "Просте речення. ЦІНА 500 ГРН."


def test_routing_can_be_switched_off() -> None:
    verbalizer = FakeVerbalizer()
    build(verbalizer, route=False).prepare("Він не має руки.")
    assert verbalizer.calls == [["Він не має руки."]]
