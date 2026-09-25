"""Stable, domain-separated content fingerprints."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel

FINGERPRINT_VERSION = "sha256-v1"


def _canonical_value(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return _canonical_value(value.model_dump(mode="json"))
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _canonical_value(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _canonical_value(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_canonical_value(item) for item in value]
    if isinstance(value, set | frozenset):
        canonical = [_canonical_value(item) for item in value]
        return sorted(canonical, key=_canonical_json)
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, Enum):
        return _canonical_value(value.value)
    if value is None or isinstance(value, str | int | float | bool):
        return value
    raise TypeError(f"unsupported fingerprint value: {type(value).__qualname__}")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_json_bytes(value: Any) -> bytes:
    """Encode structured data deterministically as UTF-8 JSON."""

    return _canonical_json(_canonical_value(value)).encode("utf-8")


def content_fingerprint(value: Any, *, namespace: str) -> str:
    """Hash structured content with an explicit domain namespace."""

    if not namespace or "\0" in namespace:
        raise ValueError("fingerprint namespace must be non-empty and cannot contain NUL")
    digest = hashlib.sha256()
    digest.update(FINGERPRINT_VERSION.encode("ascii"))
    digest.update(b"\0")
    digest.update(namespace.encode("utf-8"))
    digest.update(b"\0")
    digest.update(canonical_json_bytes(value))
    return f"sha256:{digest.hexdigest()}"


def file_fingerprint(path: str | Path, *, namespace: str = "file") -> str:
    """Hash raw file bytes without loading the entire file into memory."""

    digest = hashlib.sha256()
    digest.update(FINGERPRINT_VERSION.encode("ascii"))
    digest.update(b"\0")
    digest.update(namespace.encode("utf-8"))
    digest.update(b"\0")
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def ordered_fingerprint(values: Sequence[str], *, namespace: str) -> str:
    """Fingerprint an ordered sequence, preserving duplicates and order."""

    return content_fingerprint(list(values), namespace=namespace)
