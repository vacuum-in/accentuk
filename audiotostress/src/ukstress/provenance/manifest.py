"""Typed experiment manifest with an immutable content identity."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from ukstress.provenance.fingerprints import content_fingerprint


class ExperimentManifest(BaseModel):
    """Reproducibility metadata persisted for every train/evaluate/mine run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "1.0"
    run_id: str = Field(min_length=1)
    git_commit: str | None = None
    command: list[str]
    config_snapshot: dict[str, Any]
    config_fingerprint: str
    random_seeds: dict[str, int]
    inputs: dict[str, str]
    models: dict[str, str | None] = Field(default_factory=dict)
    alignment: dict[str, str] = Field(default_factory=dict)
    environment: dict[str, str] = Field(default_factory=dict)
    outputs: dict[str, str] = Field(default_factory=dict)

    @property
    def fingerprint(self) -> str:
        return content_fingerprint(self, namespace="experiment-manifest")

    def write(self, path: str | Path) -> None:
        """Atomically create a manifest; never overwrite an existing one."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        payload = self.model_dump(mode="json")
        payload["manifest_fingerprint"] = self.fingerprint
        try:
            with temporary.open("x", encoding="utf-8") as stream:
                yaml.safe_dump(payload, stream, allow_unicode=True, sort_keys=True)
                stream.flush()
                os.fsync(stream.fileno())
            if destination.exists():
                raise FileExistsError(f"experiment manifest already exists: {destination}")
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)

    @classmethod
    def read(cls, path: str | Path) -> ExperimentManifest:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("experiment manifest must contain a YAML mapping")
        expected = raw.pop("manifest_fingerprint", None)
        manifest = cls.model_validate(raw)
        if expected != manifest.fingerprint:
            raise ValueError("experiment manifest fingerprint mismatch")
        return manifest
