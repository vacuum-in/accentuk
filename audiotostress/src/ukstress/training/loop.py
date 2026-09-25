"""Ranker metrics, structured metric logs, and leakage-safe early stopping."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ukstress.ranker import (
    VowelBatch,
    VowelStressRanker,
    apply_candidate_mask,
    masked_cross_entropy,
)
from ukstress.ranker.model import save_ranker_checkpoint


@dataclass(frozen=True)
class RankerMetrics:
    loss: float
    masked_accuracy: float
    unmasked_accuracy: float
    examples: int


def evaluate_ranker(
    ranker: VowelStressRanker,
    batches: list[VowelBatch],
    candidate_masks: list[np.ndarray],
    *,
    device: str = "cpu",
) -> RankerMetrics:
    """Evaluate both lexicon-masked and raw acoustic vowel predictions."""

    if len(batches) != len(candidate_masks) or not batches:
        raise ValueError("batches and candidate masks must be non-empty and matching")
    torch: Any = __import__("torch")
    losses: list[float] = []
    masked_correct = 0
    unmasked_correct = 0
    count = 0
    with torch.inference_mode():
        for batch, candidate_mask in zip(batches, candidate_masks, strict=True):
            if batch.targets is None:
                raise ValueError("evaluation batches require gold vowel targets")
            logits = ranker.logits(batch, device=device)
            valid = torch.as_tensor(batch.valid_mask, device=device)
            candidates = torch.as_tensor(candidate_mask, device=device) & valid
            targets = torch.as_tensor(batch.targets, device=device)
            losses.append(float(masked_cross_entropy(logits, candidates, targets)))
            masked_correct += int(
                (apply_candidate_mask(logits, candidates).argmax(dim=-1) == targets).sum()
            )
            unmasked_correct += int(
                (apply_candidate_mask(logits, valid).argmax(dim=-1) == targets).sum()
            )
            count += len(batch.targets)
    return RankerMetrics(
        loss=sum(losses) / len(losses),
        masked_accuracy=masked_correct / count,
        unmasked_accuracy=unmasked_correct / count,
        examples=count,
    )


def append_metric_log(path: str | Path, *, split: str, step: int, metrics: RankerMetrics) -> None:
    if split not in {"train", "validation"} or step < 0:
        raise ValueError("metric logs require train/validation split and non-negative step")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {"split": split, "step": step, **asdict(metrics)}
    with destination.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")


@dataclass
class EarlyStopping:
    patience: int
    best_score: float | None = None
    non_improving_steps: int = 0

    def __post_init__(self) -> None:
        if self.patience < 1:
            raise ValueError("patience must be positive")

    def update(self, score: float) -> bool:
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            self.non_improving_steps = 0
            return True
        self.non_improving_steps += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.non_improving_steps >= self.patience


def select_validation_checkpoint(
    ranker: VowelStressRanker,
    *,
    validation_metrics: RankerMetrics,
    stopping: EarlyStopping,
    path: str | Path,
    metadata: dict[str, str],
    train_record_ids: set[str],
    validation_record_ids: set[str],
) -> bool:
    """Save only improved validation checkpoints after a leakage guard."""

    overlap = train_record_ids & validation_record_ids
    if overlap:
        raise ValueError("validation checkpoint selection requires leakage-safe splits")
    improved = stopping.update(validation_metrics.masked_accuracy)
    if improved:
        save_ranker_checkpoint(
            path,
            ranker,
            metadata={
                **metadata,
                "validation_masked_accuracy": str(validation_metrics.masked_accuracy),
            },
        )
    return improved


def train_ranker_step(
    ranker: VowelStressRanker,
    batch: VowelBatch,
    candidate_mask: np.ndarray,
    optimizer: Any,
    *,
    device: str = "cpu",
) -> float:
    """Single CPU/GPU training step used by the smoke pipeline and full trainer."""

    if batch.targets is None:
        raise ValueError("training batches require gold vowel targets")
    torch: Any = __import__("torch")
    ranker.module.train()
    optimizer.zero_grad()
    logits = ranker.logits(batch, device=device)
    mask = torch.as_tensor(candidate_mask, device=device) & torch.as_tensor(
        batch.valid_mask, device=device
    )
    loss = masked_cross_entropy(logits, mask, torch.as_tensor(batch.targets, device=device))
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu())
