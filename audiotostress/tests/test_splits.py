from ukstress.datasets import FeatureExample, VowelInterval
from ukstress.splits import balance_by_lexeme, speaker_held_out_split, target_word_held_out_split
from ukstress.splits.core import split_report


def _example(record_id: str, word: str, speaker: str) -> FeatureExample:
    return FeatureExample(
        record_id=record_id,
        source_id="s",
        utterance_id=record_id,
        speaker_id=speaker,
        audio_uri="audio.wav",
        target_word=word,
        word_start_s=0.0,
        word_end_s=0.2,
        vowels=[VowelInterval(vowel_index=0, grapheme="а", start_s=0.01, end_s=0.1)],
        gold_vowel_index=0,
        lexicon_fingerprint="lex",
        config_fingerprint="cfg",
    )


def test_group_splits_are_deterministic_and_balance_caps_lexemes() -> None:
    examples = [
        _example("a", "замок", "one"),
        _example("b", "замок", "one"),
        _example("c", "молоко", "two"),
    ]
    speakers = speaker_held_out_split(examples, seed=17, validation_fraction=0.2, test_fraction=0.2)
    words = target_word_held_out_split(
        examples, seed=17, validation_fraction=0.2, test_fraction=0.2
    )
    report = split_report(
        speakers, examples, strategy="speaker", seed=17, group=lambda item: item.speaker_id
    )

    assert sorted(record_id for values in speakers.values() for record_id in values) == [
        "a",
        "b",
        "c",
    ]
    assert sorted(record_id for values in words.values() for record_id in values) == ["a", "b", "c"]
    assert len(balance_by_lexeme(examples, maximum_per_lexeme=1)) == 2
    assert report.record_counts and report.fingerprint.startswith("sha256:")
