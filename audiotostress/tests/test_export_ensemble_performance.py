import numpy as np

from ukstress.datasets import CorpusManifestRecord, MiningRecord, StressCandidate, VowelInterval
from ukstress.datasets.parquet import write_parquet_shard
from ukstress.ensemble import EnsembleCalibrator, select_high_precision_profile
from ukstress.export import TextResolverExportPolicy, export_text_resolver
from ukstress.performance import StageCache, StageProfiler, bounded_map, pending_shards
from ukstress.provenance import derived_provenance


def _record(*, accepted: bool = True) -> MiningRecord:
    return MiningRecord(
        record_id="rec_export",
        source_id="source",
        utterance_id="utt",
        audio_uri="audio.wav",
        sentence_original="Перевірив замок.",
        sentence_normalized="перевірив замок.",
        target_word="замок",
        target_char_start=10,
        target_char_end=15,
        word_start_s=0.1,
        word_end_s=0.5,
        vowels=[
            VowelInterval(vowel_index=0, grapheme="а", start_s=0.1, end_s=0.2),
            VowelInterval(vowel_index=1, grapheme="о", start_s=0.3, end_s=0.4),
        ],
        candidates=[
            StressCandidate(stressed_form="за́мок", vowel_index=0, source="test"),
            StressCandidate(stressed_form="замо́к", vowel_index=1, source="test"),
        ],
        ranker_probs=[0.1, 0.9],
        predicted_candidate=1,
        calibrated_confidence=0.9,
        alignment_score=1.0,
        identity_score=1.0,
        accepted=accepted,
        rejection_reasons=[] if accepted else ["low_candidate_margin"],
        lexicon_fingerprint="sha256:lex",
        config_fingerprint="cfg",
        redistribution=False,
        license_id="internal",
        provenance={"corpus": "fixture"},
    )


def test_export_gate_cache_and_bounded_shards(tmp_path) -> None:
    result = export_text_resolver(
        [_record()], tmp_path / "export", policy=TextResolverExportPolicy()
    )
    assert result.record_count == 1
    assert "<w>замок</w>" in result.jsonl_path.read_text(encoding="utf-8")
    assert '"audio_uri": null' in result.jsonl_path.read_text(encoding="utf-8")
    profile = select_high_precision_profile(np.array([0.9, 0.8]), np.array([1, 0]), min_samples=1)
    assert not profile.validated
    calibrator = EnsembleCalibrator.fit(
        np.array([0.2, 0.8]), np.array([0, 1]), split_name="validation", manual_gold=True
    )
    assert calibrator.transform(np.array([0.5])).shape == (1,)
    cache = StageCache(tmp_path / "cache")
    key = cache.key("asr", input_fingerprint="sha256:i", config_fingerprint="sha256:c")
    cache.store(key, {"text": "ok"})
    assert cache.load(key) == {"text": "ok"}
    assert list(bounded_map(lambda value: value * 2, [1, 2, 3], max_in_flight=2)) == [2, 4, 6]
    profiler = StageProfiler()
    with profiler.measure("ranker", items=3):
        pass
    assert profiler.stages["ranker"]["items"] == 3


def test_pending_shards_and_license_provenance(tmp_path) -> None:
    write_parquet_shard([_record()], tmp_path, shard_id="a", input_fingerprint="sha256:input")
    assert pending_shards(tmp_path, ["a", "b"], "sha256:input") == ["b"]
    source = CorpusManifestRecord(
        source_id="source",
        utterance_id="utt",
        audio_uri="audio.wav",
        license_id="CC0",
        redistribution=True,
        provenance={"url": "https://example"},
    )
    derived = derived_provenance(source, stage="alignment")
    assert derived["license_id"] == "CC0" and derived["url"] == "https://example"
