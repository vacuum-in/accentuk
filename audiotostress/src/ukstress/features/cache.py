"""Atomic deterministic cache for extracted SSL feature matrices."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ukstress.provenance.fingerprints import content_fingerprint


def feature_cache_key(*, encoder_id: str, config_fingerprint: str, input_fingerprint: str) -> str:
    return content_fingerprint(
        {
            "encoder_id": encoder_id,
            "config_fingerprint": config_fingerprint,
            "input_fingerprint": input_fingerprint,
        },
        namespace="ssl-feature-cache",
    )


@dataclass(frozen=True)
class CachedFeatureArtifact:
    embeddings: np.ndarray
    metadata: dict[str, str]


class FeatureCache:
    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)

    def path_for(self, key: str) -> Path:
        if not key.startswith("sha256:"):
            raise ValueError("feature cache key must be a sha256 fingerprint")
        return self.directory / f"{key.removeprefix('sha256:')}.npz"

    def load(self, key: str) -> CachedFeatureArtifact | None:
        path = self.path_for(key)
        if not path.is_file():
            return None
        with np.load(path, allow_pickle=False) as payload:
            embeddings = np.asarray(payload["embeddings"], dtype=np.float32)
            metadata = json.loads(str(payload["metadata"].item()))
        if embeddings.ndim != 2 or not isinstance(metadata, dict):
            raise ValueError(f"invalid cached feature artifact: {path}")
        normalized_metadata = {str(key): str(value) for key, value in metadata.items()}
        return CachedFeatureArtifact(embeddings=embeddings, metadata=normalized_metadata)

    def store(self, key: str, embeddings: np.ndarray, *, metadata: dict[str, str]) -> Path:
        if embeddings.ndim != 2 or not embeddings.shape[0]:
            raise ValueError("embeddings must be a non-empty [frames, features] matrix")
        destination = self.path_for(key)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        try:
            with temporary.open("wb") as stream:
                np.savez_compressed(
                    stream,
                    embeddings=np.asarray(embeddings, dtype=np.float32),
                    metadata=json.dumps(metadata, ensure_ascii=False, sort_keys=True),
                )
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        return destination
