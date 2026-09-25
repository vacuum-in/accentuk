"""Deterministic random seed utilities."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import os
import random
from typing import Any

import numpy as np

MAX_SEED = 2**32 - 1


def validate_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= MAX_SEED:
        raise ValueError(f"seed must be an integer in [0, {MAX_SEED}]")
    return seed


def derive_seed(seed: int, *, namespace: str, index: int = 0) -> int:
    """Derive a stable independent 32-bit seed for a worker or component."""

    validate_seed(seed)
    if not namespace:
        raise ValueError("seed namespace must be non-empty")
    if index < 0:
        raise ValueError("seed index must be non-negative")
    digest = hashlib.sha256(f"ukstress-seed-v1\0{seed}\0{namespace}\0{index}".encode()).digest()
    return int.from_bytes(digest[:4], byteorder="big")


def seed_everything(seed: int, *, deterministic_torch: bool = True) -> dict[str, int]:
    """Seed installed random backends and return manifest-ready seed metadata."""

    seed = validate_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    applied = {"python": seed, "numpy": seed}

    if importlib.util.find_spec("torch") is not None:
        torch: Any = importlib.import_module("torch")
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.benchmark = False
        applied["torch"] = seed
    return applied
