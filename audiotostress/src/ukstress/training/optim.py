"""Optimizer, scheduler, clipping, and precision configuration."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TrainingOptions:
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    max_grad_norm: float = 1.0
    scheduler: str = "cosine"
    mixed_precision: bool = False

    def __post_init__(self) -> None:
        if self.learning_rate <= 0.0 or self.max_grad_norm <= 0.0 or self.weight_decay < 0.0:
            raise ValueError(
                "training hyperparameters must be non-negative with positive learning/clipping"
            )
        if self.scheduler not in {"cosine", "none"}:
            raise ValueError("scheduler must be cosine or none")


def build_optimizer_scheduler(
    parameters: Any, options: TrainingOptions, *, total_steps: int
) -> tuple[Any, Any | None]:
    if total_steps < 1:
        raise ValueError("total_steps must be positive")
    try:
        torch = importlib.import_module("torch")
    except ImportError as error:
        raise RuntimeError("training requires `uv sync --extra training`") from error
    optimizer = torch.optim.AdamW(
        parameters, lr=options.learning_rate, weight_decay=options.weight_decay
    )
    scheduler = (
        None
        if options.scheduler == "none"
        else torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    )
    return optimizer, scheduler


def clip_gradients(parameters: Any, options: TrainingOptions) -> float:
    torch = importlib.import_module("torch")
    return float(torch.nn.utils.clip_grad_norm_(parameters, options.max_grad_norm))
