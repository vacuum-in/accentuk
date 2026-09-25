"""Small deterministic scaling primitives that do not prescribe a scheduler."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeVar

from ukstress.datasets.parquet import completed_shards
from ukstress.provenance.fingerprints import content_fingerprint

T = TypeVar("T")
U = TypeVar("U")


class StageCache:
    """Generic deterministic JSON cache shared by ASR, alignment, and feature stages."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)

    def key(self, stage: str, *, input_fingerprint: str, config_fingerprint: str) -> str:
        return content_fingerprint(
            {"stage": stage, "input": input_fingerprint, "config": config_fingerprint},
            namespace="stage-cache",
        )

    def load(self, key: str) -> dict[str, object] | None:
        path = self._path(key)
        if not path.is_file():
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"cached stage artifact must be an object: {path}")
        return value

    def store(self, key: str, value: dict[str, object]) -> Path:
        destination = self._path(key)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        try:
            temporary.write_text(
                json.dumps(value, ensure_ascii=False, sort_keys=True), encoding="utf-8"
            )
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        return destination

    def _path(self, key: str) -> Path:
        if not key.startswith("sha256:"):
            raise ValueError("stage cache key must be a sha256 fingerprint")
        return self.directory / f"{key.removeprefix('sha256:')}.json"


@dataclass
class StageProfiler:
    """Record elapsed time and item counts independently for each pipeline stage."""

    stages: dict[str, dict[str, float | int]] = field(default_factory=dict)

    def measure(self, stage: str, *, items: int = 0) -> _StageTimer:
        if not stage.strip() or items < 0:
            raise ValueError("stage must be non-empty and items must be non-negative")
        return _StageTimer(self, stage, items)

    def record(self, stage: str, elapsed_s: float, *, items: int = 0) -> None:
        current = self.stages.setdefault(stage, {"elapsed_s": 0.0, "items": 0})
        current["elapsed_s"] = float(current["elapsed_s"]) + elapsed_s
        current["items"] = int(current["items"]) + items


@dataclass
class _StageTimer:
    profiler: StageProfiler
    stage: str
    items: int
    started: float = field(default_factory=time.perf_counter)

    def __enter__(self) -> _StageTimer:
        return self

    def __exit__(self, *_: object) -> None:
        self.profiler.record(self.stage, time.perf_counter() - self.started, items=self.items)


def bounded_map(
    function: Callable[[T], U], values: Iterable[T], *, max_in_flight: int = 8
) -> Iterator[U]:
    """Apply work with bounded outstanding futures to provide backpressure."""

    if max_in_flight < 1:
        raise ValueError("max_in_flight must be positive")
    iterator = iter(values)
    with ThreadPoolExecutor(max_workers=max_in_flight) as executor:
        futures: list[Future[U]] = []
        for _ in range(max_in_flight):
            try:
                futures.append(executor.submit(function, next(iterator)))
            except StopIteration:
                break
        while futures:
            future = futures.pop(0)
            yield future.result()
            with suppress(StopIteration):
                futures.append(executor.submit(function, next(iterator)))


def run_shards(
    shard_ids: Sequence[str], worker: Callable[[str], U], *, max_workers: int = 1
) -> dict[str, U]:
    """Run shard workers in stable ID order; callers can skip completed shards in the worker."""

    if max_workers < 1:
        raise ValueError("max_workers must be positive")
    unique = sorted(set(shard_ids))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {shard_id: executor.submit(worker, shard_id) for shard_id in unique}
        return {shard_id: futures[shard_id].result() for shard_id in unique}


def pending_shards(
    output_dir: str | Path, shard_ids: Sequence[str], input_fingerprint: str
) -> list[str]:
    """Return only uncommitted/conflicting shard IDs for restart-safe processing."""

    completed = completed_shards(output_dir)
    return [
        shard_id
        for shard_id in sorted(set(shard_ids))
        if shard_id not in completed or completed[shard_id].input_fingerprint != input_fingerprint
    ]
