import pytest

from ukstress.datasets import FeatureExample, VowelInterval
from ukstress.datasets.parquet import write_parquet_shard
from ukstress.training import (
    BootstrapFeatureDataset,
    TrainingOptions,
    build_optimizer_scheduler,
    make_bootstrap_dataloader,
)


def _feature() -> FeatureExample:
    return FeatureExample(
        record_id="rec",
        source_id="s",
        utterance_id="u",
        audio_uri="audio.wav",
        target_word="молоко",
        word_start_s=0.0,
        word_end_s=0.2,
        vowels=[VowelInterval(vowel_index=0, grapheme="о", start_s=0.01, end_s=0.1)],
        gold_vowel_index=0,
        ssl_features=[[0.1, 0.2]],
        prosodic_features=[{"duration_s": 0.1, "f0_valid": True}],
        lexicon_fingerprint="lex",
        config_fingerprint="cfg",
    )


def test_bootstrap_parquet_dataset_and_training_options(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    shard = write_parquet_shard(
        [_feature()], tmp_path, shard_id="1", input_fingerprint="sha256:input"
    )
    dataset = BootstrapFeatureDataset([shard.status.path])
    loader = make_bootstrap_dataloader(dataset, batch_size=1, shuffle=False, seed=17)
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer, scheduler = build_optimizer_scheduler([parameter], TrainingOptions(), total_steps=2)

    assert len(dataset) == 1 and next(iter(loader))[0].record_id == "rec"
    assert optimizer.param_groups[0]["lr"] > 0 and scheduler is not None
