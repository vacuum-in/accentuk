import json

import numpy as np
import pytest

from ukstress.ranker import VowelStressRanker, make_vowel_batch
from ukstress.training import (
    EarlyStopping,
    append_metric_log,
    evaluate_ranker,
    select_validation_checkpoint,
)


def _batch():
    return make_vowel_batch(
        [np.ones((2, 2), dtype=np.float32)],
        [np.ones((2, 1), dtype=np.float32)],
        targets=[0],
    )


def test_metrics_logging_and_leakage_safe_early_stopping(tmp_path) -> None:
    pytest.importorskip("torch")
    ranker = VowelStressRanker(
        ssl_size=2,
        prosody_size=1,
        hidden_size=4,
        transformer_layers=2,
        attention_heads=2,
        ffn_size=8,
    )
    metrics = evaluate_ranker(ranker, [_batch()], [np.array([[True, True]])])
    log = tmp_path / "metrics.jsonl"
    append_metric_log(log, split="validation", step=2, metrics=metrics)
    stopping = EarlyStopping(patience=2)
    selected = select_validation_checkpoint(
        ranker,
        validation_metrics=metrics,
        stopping=stopping,
        path=tmp_path / "best.pt",
        metadata={"config": "cfg"},
        train_record_ids={"train"},
        validation_record_ids={"validation"},
    )

    assert json.loads(log.read_text(encoding="utf-8"))["unmasked_accuracy"] >= 0.0
    assert selected and (tmp_path / "best.pt").is_file()
    with pytest.raises(ValueError, match="leakage-safe"):
        select_validation_checkpoint(
            ranker,
            validation_metrics=metrics,
            stopping=stopping,
            path=tmp_path / "bad.pt",
            metadata={},
            train_record_ids={"same"},
            validation_record_ids={"same"},
        )
