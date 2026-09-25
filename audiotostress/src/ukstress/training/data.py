"""Read bootstrap FeatureExamples from canonical Parquet shards."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from ukstress.datasets import FeatureExample


class BootstrapFeatureDataset:
    def __init__(self, paths: list[str | Path]) -> None:
        self.examples: list[FeatureExample] = []
        for path in paths:
            table = pq.read_table(path)
            for row in table.to_pylist():
                prosody = row.pop("prosodic_features_json", [])
                row["prosodic_features"] = [json.loads(value) for value in prosody]
                row["provenance"] = dict(row.get("provenance") or [])
                self.examples.append(FeatureExample.model_validate(row))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> FeatureExample:
        return self.examples[index]


def make_bootstrap_dataloader(
    dataset: BootstrapFeatureDataset, *, batch_size: int, shuffle: bool, seed: int
) -> Any:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    try:
        torch = importlib.import_module("torch")
    except ImportError as error:
        raise RuntimeError("training requires `uv sync --extra training`") from error
    generator = torch.Generator().manual_seed(seed)
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, generator=generator, collate_fn=list
    )
