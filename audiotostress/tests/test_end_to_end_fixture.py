import numpy as np
import soundfile as sf

from ukstress.alignment import FineAlignmentResult
from ukstress.alignment.interface import WordAlignmentResult
from ukstress.datasets import (
    CorpusManifestRecord,
    FeatureExample,
    VowelInterval,
    WordAlignment,
)
from ukstress.datasets.parquet import write_parquet_shard
from ukstress.features import extract_prosodic_features
from ukstress.pipeline.ingest import ingest_record
from ukstress.ranker import VowelStressRanker, make_vowel_batch


def test_tiny_fixture_ingest_alignment_features_inference_parquet(tmp_path) -> None:
    audio_path = tmp_path / "fixture.wav"
    waveform = np.sin(np.linspace(0, 30, 16_000)).astype(np.float32)
    sf.write(audio_path, waveform, 16_000)
    manifest = CorpusManifestRecord(
        source_id="fixture", utterance_id="u1", audio_uri=str(audio_path), transcript="замок"
    )
    ingested = ingest_record(manifest)
    assert ingested.utterance is not None and ingested.audio is not None
    word = WordAlignment(
        token="замок",
        normalized_token="замок",
        start_s=0.1,
        end_s=0.8,
        backend="mock",
        model_id="mock-v1",
    )
    alignment = WordAlignmentResult(
        backend="mock", model_id="mock-v1", utterance_duration_s=1.0, words=[word], quality=1.0
    )
    fine = FineAlignmentResult(
        backend="mock",
        model_id="mock-fine",
        target_word="замок",
        word_start_s=0.1,
        word_end_s=0.8,
        quality=1.0,
        vowels=[
            VowelInterval(vowel_index=0, grapheme="а", start_s=0.2, end_s=0.35),
            VowelInterval(vowel_index=1, grapheme="о", start_s=0.5, end_s=0.65),
        ],
    )
    prosody = extract_prosodic_features(
        ingested.audio.waveform, 16_000, fine.vowels, word_start_s=0.1, word_end_s=0.8
    )
    example = FeatureExample(
        record_id="fixture-record",
        source_id="fixture",
        utterance_id="u1",
        audio_uri=str(audio_path),
        target_word="замок",
        word_start_s=0.1,
        word_end_s=0.8,
        vowels=fine.vowels,
        gold_vowel_index=0,
        ssl_features=[[0.1, 0.2], [0.2, 0.1]],
        prosodic_features=prosody,
        lexicon_fingerprint="sha256:lex",
        config_fingerprint="sha256:cfg",
    )
    ranker = VowelStressRanker(
        ssl_size=2, prosody_size=len(prosody[0]), hidden_size=4, attention_heads=2
    )
    batch = make_vowel_batch(
        [np.asarray(example.ssl_features, dtype=np.float32)],
        [
            np.asarray(
                [[float(value or 0.0) for value in item.values()] for item in prosody],
                dtype=np.float32,
            )
        ],
    )
    probabilities = ranker.probabilities(batch)[0]
    assert np.isclose(probabilities.sum(), 1.0)
    written = write_parquet_shard(
        [example], tmp_path / "features", shard_id="0001", input_fingerprint="sha256:fixture"
    )
    assert written.status.record_count == 1 and alignment.words[0].normalized_token == "замок"
