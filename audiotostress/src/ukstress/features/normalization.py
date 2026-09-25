"""Training-split-only normalization for continuous prosodic features."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ProsodyNormalizer:
    means: dict[str, float]
    standard_deviations: dict[str, float]
    training_record_ids: tuple[str, ...]

    @classmethod
    def fit(
        cls,
        feature_rows: list[dict[str, float | bool | None]],
        *,
        split_name: str,
        training_record_ids: list[str],
    ) -> ProsodyNormalizer:
        if split_name != "train":
            raise ValueError("prosody normalization may be fit only on the training split")
        if not training_record_ids:
            raise ValueError("training_record_ids must be non-empty")
        values: dict[str, list[float]] = {}
        for row in feature_rows:
            for name, value in row.items():
                if isinstance(value, bool) or value is None or not isinstance(value, (float, int)):
                    continue
                if math.isfinite(float(value)):
                    values.setdefault(name, []).append(float(value))
        means: dict[str, float] = {}
        deviations: dict[str, float] = {}
        for name, column in values.items():
            mean = sum(column) / len(column)
            variance = sum((item - mean) ** 2 for item in column) / len(column)
            means[name] = mean
            deviations[name] = math.sqrt(variance) or 1.0
        return cls(means, deviations, tuple(sorted(training_record_ids)))

    def transform(self, row: dict[str, float | bool | None]) -> dict[str, float | bool | None]:
        output = dict(row)
        for name, mean in self.means.items():
            value = output.get(name)
            if isinstance(value, bool) or value is None or not isinstance(value, (float, int)):
                continue
            output[name] = (float(value) - mean) / self.standard_deviations[name]
        return output

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        payload = {
            "means": self.means,
            "standard_deviations": self.standard_deviations,
            "training_record_ids": list(self.training_record_ids),
        }
        try:
            temporary.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        return destination

    @classmethod
    def load(cls, path: str | Path) -> ProsodyNormalizer:
        payload: Any = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            means={str(key): float(value) for key, value in payload["means"].items()},
            standard_deviations={
                str(key): float(value) for key, value in payload["standard_deviations"].items()
            },
            training_record_ids=tuple(str(value) for value in payload["training_record_ids"]),
        )
