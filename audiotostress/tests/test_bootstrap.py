from typing import ClassVar

from ukstress.alignment import FineAlignmentResult
from ukstress.alignment.interface import WordAlignmentResult
from ukstress.config.models import AlignmentConfig
from ukstress.datasets import NormalizedUtterance, StressCandidate, VowelInterval, WordAlignment
from ukstress.pipeline.bootstrap import build_bootstrap_examples, write_bootstrap_shard


class FixtureLexicon:
    version = "fixture-1"
    fingerprint = "sha256:lexicon"
    provenance: ClassVar[dict[str, str]] = {"source": "fixture"}

    def lookup(self, surface: str) -> tuple[StressCandidate, ...]:
        entries = {
            "працівник": (
                StressCandidate(stressed_form="пра́цівник", vowel_index=0, source="fixture"),
            ),
            "замок": (
                StressCandidate(stressed_form="за́мок", vowel_index=0, source="fixture"),
                StressCandidate(stressed_form="замо́к", vowel_index=1, source="fixture"),
            ),
            "це": (StressCandidate(stressed_form="це́", vowel_index=0, source="fixture"),),
        }
        return entries.get(surface, ())


def _utterance() -> NormalizedUtterance:
    return NormalizedUtterance(
        record_id="utt_1",
        source_id="source",
        utterance_id="utterance",
        audio_uri="audio.wav",
        transcript_original="Працівник перевірив замок це",
        transcript_normalized="працівник перевірив замок це",
        sample_rate=16_000,
        duration_s=2.0,
        source_sample_rate=16_000,
        source_channels=1,
    )


def test_bootstrap_accepts_only_unique_multivowel_entries_and_tracks_rejections() -> None:
    words = [
        WordAlignment(
            token="працівник", normalized_token="працівник", start_s=0.1, end_s=0.7,
            backend="wx", model_id="m"
        ),
        WordAlignment(
            token="замок", normalized_token="замок", start_s=0.8, end_s=1.2,
            backend="wx", model_id="m"
        ),
        WordAlignment(
            token="невідомо", normalized_token="невідомо", start_s=1.3, end_s=1.6,
            backend="wx", model_id="m"
        ),
        WordAlignment(
            token="це", normalized_token="це", start_s=1.7, end_s=1.8,
            backend="wx", model_id="m"
        ),
    ]
    result = WordAlignmentResult(
        backend="wx", model_id="m", utterance_duration_s=2.0, words=words, quality=1.0
    )
    fine = FineAlignmentResult(
        backend="mfa",
        model_id="mfa-model",
        target_word="працівник",
        word_start_s=0.1,
        word_end_s=0.7,
        vowels=[
            VowelInterval(vowel_index=0, grapheme="а", start_s=0.2, end_s=0.3),
            VowelInterval(vowel_index=1, grapheme="і", start_s=0.4, end_s=0.5),
            VowelInterval(vowel_index=2, grapheme="и", start_s=0.55, end_s=0.6),
        ],
        quality=1.0,
    )
    examples, stats = build_bootstrap_examples(
        _utterance(), result, {0: fine}, FixtureLexicon(), config_fingerprint="sha256:config"
    )

    assert [example.target_word for example in examples] == ["працівник"]
    assert examples[0].gold_vowel_index == 0
    assert examples[0].provenance["fine_alignment_backend"] == "mfa"
    assert stats.total_words == 4
    assert stats.accepted == 1
    assert stats.skipped == 3
    assert stats.ambiguous == 1
    assert stats.oov == 1
    assert stats.one_vowel == 1


def test_bootstrap_can_include_one_vowel_words_when_explicitly_requested() -> None:
    word = WordAlignment(token="це", normalized_token="це", start_s=0.1, end_s=0.3, backend="wx")
    alignment = WordAlignmentResult(backend="wx", utterance_duration_s=1.0, words=[word])
    fine = FineAlignmentResult(
        backend="mfa",
        target_word="це",
        word_start_s=0.1,
        word_end_s=0.3,
        vowels=[VowelInterval(vowel_index=0, grapheme="е", start_s=0.15, end_s=0.2)],
        quality=1.0,
    )
    examples, stats = build_bootstrap_examples(
        _utterance(),
        alignment,
        {0: fine},
        FixtureLexicon(),
        config_fingerprint="config",
        alignment_config=AlignmentConfig(min_quality=0.9),
        exclude_one_vowel=False,
    )

    assert len(examples) == 1
    assert stats.accepted == 1


def test_bootstrap_caps_repeated_lexemes_using_shared_deterministic_counts() -> None:
    word = WordAlignment(
        token="працівник", normalized_token="працівник", start_s=0.1, end_s=0.7, backend="wx"
    )
    alignment = WordAlignmentResult(backend="wx", utterance_duration_s=1.0, words=[word])
    fine = FineAlignmentResult(
        backend="mfa",
        target_word="працівник",
        word_start_s=0.1,
        word_end_s=0.7,
        vowels=[
            VowelInterval(vowel_index=0, grapheme="а", start_s=0.2, end_s=0.3),
            VowelInterval(vowel_index=1, grapheme="і", start_s=0.4, end_s=0.5),
            VowelInterval(vowel_index=2, grapheme="и", start_s=0.55, end_s=0.6),
        ],
        quality=1.0,
    )
    counts: dict[str, int] = {}
    first, first_stats = build_bootstrap_examples(
        _utterance(), alignment, {0: fine}, FixtureLexicon(),
        config_fingerprint="config", occurrence_counts=counts, max_occurrences_per_lexeme=1,
    )
    second, second_stats = build_bootstrap_examples(
        _utterance(), alignment, {0: fine}, FixtureLexicon(),
        config_fingerprint="config", occurrence_counts=counts, max_occurrences_per_lexeme=1,
    )

    assert len(first) == 1 and first_stats.accepted == 1
    assert second == [] and second_stats.capped == 1 and second_stats.skipped == 1


def test_bootstrap_writer_uses_atomic_canonical_feature_schema(tmp_path) -> None:
    word = WordAlignment(
        token="працівник", normalized_token="працівник", start_s=0.1, end_s=0.7, backend="wx"
    )
    alignment = WordAlignmentResult(backend="wx", utterance_duration_s=1.0, words=[word])
    fine = FineAlignmentResult(
        backend="mfa", target_word="працівник", word_start_s=0.1, word_end_s=0.7,
        vowels=[
            VowelInterval(vowel_index=0, grapheme="а", start_s=0.2, end_s=0.3),
            VowelInterval(vowel_index=1, grapheme="і", start_s=0.4, end_s=0.5),
            VowelInterval(vowel_index=2, grapheme="и", start_s=0.55, end_s=0.6),
        ], quality=1.0,
    )
    examples, _ = build_bootstrap_examples(
        _utterance(), alignment, {0: fine}, FixtureLexicon(), config_fingerprint="config"
    )

    written = write_bootstrap_shard(
        examples, str(tmp_path), shard_id="0001", input_fingerprint="sha256:input"
    )
    assert written.status.record_type == "feature_example"
    assert written.status.record_count == 1
    assert written.skipped is False
