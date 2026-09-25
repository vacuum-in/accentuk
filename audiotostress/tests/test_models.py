import pytest
from pydantic import ValidationError

from ukstress.datasets import (
    CorpusManifestRecord,
    FeatureExample,
    MiningRecord,
    StressCandidate,
    VowelInterval,
    WordAlignment,
)


def _vowels() -> list[VowelInterval]:
    return [
        VowelInterval(vowel_index=0, grapheme="а", start_s=1.0, end_s=1.1),
        VowelInterval(vowel_index=1, grapheme="о", start_s=1.2, end_s=1.3),
    ]


def _candidates() -> list[StressCandidate]:
    return [
        StressCandidate(stressed_form="за́мок", vowel_index=0, source="fixture"),
        StressCandidate(stressed_form="замо́к", vowel_index=1, source="fixture"),
    ]


def test_manifest_record_requires_complete_segment() -> None:
    with pytest.raises(ValidationError, match="specified together"):
        CorpusManifestRecord(
            source_id="source", utterance_id="utt", audio_uri="audio.wav", segment_start_s=1.0
        )


def test_word_alignment_rejects_reversed_bounds() -> None:
    with pytest.raises(ValidationError, match="start must be before end"):
        WordAlignment(
            token="замок",
            normalized_token="замок",
            start_s=2.0,
            end_s=1.0,
            backend="mock",
        )


def test_feature_example_requires_one_feature_vector_per_vowel() -> None:
    with pytest.raises(ValidationError, match="one vector per vowel"):
        FeatureExample(
            record_id="rec",
            source_id="source",
            utterance_id="utt",
            audio_uri="audio.wav",
            target_word="замок",
            word_start_s=1.0,
            word_end_s=1.3,
            vowels=_vowels(),
            gold_vowel_index=1,
            ssl_features=[[0.1]],
            lexicon_fingerprint="sha256:lexicon",
            config_fingerprint="sha256:config",
        )


def test_mining_record_preserves_complete_audit_evidence() -> None:
    record = MiningRecord(
        record_id="rec",
        source_id="source",
        utterance_id="utt",
        audio_uri="audio.wav",
        sentence_original="Працівник перевірив замок.",
        sentence_normalized="працівник перевірив замок",
        target_word="замок",
        word_start_s=1.0,
        word_end_s=1.3,
        vowels=_vowels(),
        candidates=_candidates(),
        ranker_probs=[0.01, 0.99],
        predicted_candidate=1,
        calibrated_confidence=0.995,
        accepted=True,
        lexicon_fingerprint="sha256:lexicon",
        model_versions={"ranker": "checkpoint-1"},
        config_fingerprint="sha256:config",
        provenance={"corpus": "fixture"},
    )

    assert record.candidates[record.predicted_candidate or 0].stressed_form == "замо́к"


def test_rejected_mining_record_requires_reason() -> None:
    with pytest.raises(ValidationError, match="at least one rejection reason"):
        MiningRecord(
            record_id="rec",
            source_id="source",
            utterance_id="utt",
            audio_uri="audio.wav",
            sentence_original="замок",
            sentence_normalized="замок",
            target_word="замок",
            word_start_s=1.0,
            word_end_s=1.3,
            vowels=_vowels(),
            candidates=_candidates(),
            accepted=False,
            lexicon_fingerprint="sha256:lexicon",
            config_fingerprint="sha256:config",
        )
