"""Persist a complete, fingerprinted training artifact bundle."""

from __future__ import annotations

import json
import os
from pathlib import Path

from ukstress.provenance.fingerprints import file_fingerprint
from ukstress.provenance.manifest import ExperimentManifest
from ukstress.training.loop import RankerMetrics


def save_training_artifacts(
    directory: str | Path,
    *,
    manifest: ExperimentManifest,
    train_metrics: RankerMetrics,
    validation_metrics: RankerMetrics,
    normalization_path: str | Path,
    checkpoint_path: str | Path,
) -> dict[str, str]:
    """Write manifest/metrics and bind every artifact by content fingerprint."""

    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.yaml"
    manifest.write(manifest_path)
    metrics_path = root / "metrics.json"
    temporary = metrics_path.with_name(f".{metrics_path.name}.{os.getpid()}.tmp")
    payload = {"train": train_metrics.__dict__, "validation": validation_metrics.__dict__}
    try:
        temporary.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        os.replace(temporary, metrics_path)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "manifest": file_fingerprint(manifest_path, namespace="training-artifact"),
        "metrics": file_fingerprint(metrics_path, namespace="training-artifact"),
        "normalization": file_fingerprint(normalization_path, namespace="training-artifact"),
        "checkpoint": file_fingerprint(checkpoint_path, namespace="training-artifact"),
    }
