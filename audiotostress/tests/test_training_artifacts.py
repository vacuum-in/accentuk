import numpy as np
import pytest

from ukstress.provenance import ExperimentManifest
from ukstress.ranker import VowelStressRanker, make_vowel_batch, save_ranker_checkpoint
from ukstress.training import (
    RankerMetrics,
    save_training_artifacts,
    train_ranker_step,
)


def test_cpu_training_smoke_and_artifact_bundle(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    ranker = VowelStressRanker(
        ssl_size=2,
        prosody_size=1,
        hidden_size=4,
        transformer_layers=2,
        attention_heads=2,
        ffn_size=8,
    )
    optimizer = torch.optim.AdamW(ranker.parameters(), lr=1e-3)
    batch = make_vowel_batch(
        [np.ones((2, 2), dtype=np.float32)], [np.ones((2, 1), dtype=np.float32)], targets=[0]
    )
    loss = train_ranker_step(ranker, batch, np.array([[True, True]]), optimizer)
    checkpoint = save_ranker_checkpoint(tmp_path / "ranker.pt", ranker, metadata={"config": "c"})
    normalization = tmp_path / "norm.json"
    normalization.write_text("{}", encoding="utf-8")
    manifest = ExperimentManifest(
        run_id="run",
        command=["train"],
        config_snapshot={},
        config_fingerprint="cfg",
        random_seeds={"python": 1},
        inputs={},
    )
    fingerprints = save_training_artifacts(
        tmp_path / "artifacts",
        manifest=manifest,
        train_metrics=RankerMetrics(loss, 1.0, 1.0, 1),
        validation_metrics=RankerMetrics(loss, 1.0, 1.0, 1),
        normalization_path=normalization,
        checkpoint_path=checkpoint,
    )

    assert loss > 0.0 and all(value.startswith("sha256:") for value in fingerprints.values())
