import numpy as np
import pytest

from ukstress.ranker import (
    VowelStressRanker,
    apply_candidate_mask,
    load_ranker_checkpoint,
    make_vowel_batch,
    masked_cross_entropy,
    save_ranker_checkpoint,
)


def test_ranker_batches_sequences_masks_candidates_and_round_trips_checkpoint(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    batch = make_vowel_batch(
        [np.ones((2, 3), dtype=np.float32), np.ones((1, 3), dtype=np.float32)],
        [np.ones((2, 2), dtype=np.float32), np.ones((1, 2), dtype=np.float32)],
        targets=[1, 0],
    )
    ranker = VowelStressRanker(
        ssl_size=3,
        prosody_size=2,
        hidden_size=4,
        transformer_layers=2,
        attention_heads=2,
        ffn_size=8,
    )
    logits = ranker.logits(batch)
    mask = torch.tensor([[True, True], [True, False]])
    loss = masked_cross_entropy(logits, mask, torch.tensor([1, 0]))
    probabilities = ranker.probabilities(batch, candidate_mask=mask.numpy())
    checkpoint = save_ranker_checkpoint(
        tmp_path / "ranker.pt", ranker, metadata={"config": "sha256:c"}
    )
    loaded, metadata = load_ranker_checkpoint(checkpoint)

    assert logits.shape == (2, 2)
    assert loss.item() > 0.0
    assert probabilities[1, 1] == 0.0
    assert loaded.architecture == ranker.architecture and metadata["config"] == "sha256:c"


def test_candidate_mask_rejects_empty_words() -> None:
    torch = pytest.importorskip("torch")
    with pytest.raises(ValueError, match="eligible"):
        apply_candidate_mask(torch.zeros((1, 2)), torch.tensor([[False, False]]))
