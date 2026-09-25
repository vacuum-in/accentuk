import numpy as np

from ukstress.alignment import StressCTCAdapter
from ukstress.alignment.fine_interface import FineAlignmentResult
from ukstress.alignment.interface import WordAlignmentResult
from ukstress.datasets import NormalizedUtterance, StressCandidate, WordAlignment
from ukstress.lexicon.loader import InMemoryStressLexicon
from ukstress.mining import (
    AcceptanceEvidence,
    AcceptancePolicy,
    build_mining_record,
    discover_ambiguous_candidates,
    mining_statistics,
)
from ukstress.scorer import CandidateConditionedScorer, compare_scorers


def _lexicon() -> InMemoryStressLexicon:
    return InMemoryStressLexicon(
        {
            "замок": [
                StressCandidate(stressed_form="за́мок", vowel_index=0, source="test"),
                StressCandidate(stressed_form="замо́к", vowel_index=1, source="test"),
            ]
        },
        version="test",
        fingerprint="sha256:test",
    )


def test_discovery_policy_and_scorer() -> None:
    word = WordAlignment(
        token="замок",
        normalized_token="замок",
        start_s=0.1,
        end_s=0.5,
        char_start=0,
        char_end=5,
        backend="mock",
        model_id="m",
    )
    alignment = WordAlignmentResult(
        backend="mock", model_id="m", utterance_duration_s=1.0, words=[word], quality=1.0
    )
    candidate = discover_ambiguous_candidates(alignment, _lexicon())[0]
    assert len(candidate.candidates) == 2
    decision = AcceptancePolicy(min_margin=0.12).evaluate(AcceptanceEvidence(0.8, 0.11, 1.0, 1.0))
    assert not decision.accepted and "low_candidate_margin" in decision.rejection_reasons
    result = CandidateConditionedScorer().score(np.ones(320, dtype=np.float32), ["за́мок", "замо́к"])
    assert result.probabilities.shape == (2,)
    assert np.isclose(result.probabilities.sum(), 1.0)
    assert compare_scorers([0, 1], [0, 0], [0, 1])["error_overlap"] == 0


def test_ctc_mapping_and_mining_record() -> None:
    lexicon = _lexicon()
    adapter = StressCTCAdapter(lambda *_: "замо́к", model_id="stress-ctc-v1")
    vote = adapter.predict(np.zeros(160, dtype=np.float32), 16000, "замок", lexicon.lookup("замок"))
    assert vote.candidate_index == 1
    utterance = NormalizedUtterance(
        record_id="u",
        source_id="source",
        utterance_id="utt",
        audio_uri="a.wav",
        transcript_original="Перевірив замок.",
        transcript_normalized="перевірив замок.",
        sample_rate=16000,
        duration_s=1.0,
        source_sample_rate=16000,
        source_channels=1,
    )
    fine = FineAlignmentResult(
        backend="mock",
        model_id="f",
        target_word="замок",
        word_start_s=0.1,
        word_end_s=0.5,
        quality=1.0,
        vowels=[
            {"vowel_index": 0, "grapheme": "а", "start_s": 0.1, "end_s": 0.2},
            {"vowel_index": 1, "grapheme": "о", "start_s": 0.3, "end_s": 0.4},
        ],
    )
    record = build_mining_record(
        utterance,
        discover_ambiguous_candidates(
            WordAlignmentResult(
                backend="mock",
                model_id="m",
                utterance_duration_s=1.0,
                words=[
                    WordAlignment(
                        token="замок",
                        normalized_token="замок",
                        start_s=0.1,
                        end_s=0.5,
                        char_start=9,
                        char_end=14,
                        backend="mock",
                        model_id="m",
                    )
                ],
                quality=1.0,
            ),
            lexicon,
        )[0],
        fine,
        lexicon=lexicon,
        config_fingerprint="cfg",
        ranker_probs=[0.2, 0.8],
        policy=AcceptancePolicy(min_confidence=0.7),
        identity_quality=1.0,
        stress_ctc_candidate=1,
        model_versions={"ranker": "r1"},
        calibration_version="cal1",
    )
    assert record.accepted and record.model_versions["calibration"] == "cal1"
    assert mining_statistics([record])["accepted"] == 1
