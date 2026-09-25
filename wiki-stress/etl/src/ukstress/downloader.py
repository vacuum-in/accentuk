"""Safe, resumable acquisition of a Wiktionary dump."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import Request, urlopen

logger = logging.getLogger("ukstress.downloader")


@dataclass(frozen=True)
class DumpManifest:
    url: str
    filename: str
    sha256: str
    acquired_at: str
    byte_size: int


class ChecksumMismatchError(ValueError):
    """Raised when a completed dump does not match its expected SHA-256."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_dump(url: str, destination: Path, expected_sha256: str | None = None) -> DumpManifest:
    """Resume a download into a sibling partial file, then atomically publish it."""
    start = time.monotonic()
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    logger.info("download started", extra={"url": url, "resume_offset_bytes": offset})
    headers = {"User-Agent": "ukstress-etl/0.1 (Wiktionary stress lexicon builder)"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = Request(url, headers=headers)

    with urlopen(request) as response:  # noqa: S310 -- caller controls the configured dump URL.
        append = offset > 0 and response.status == 206
        mode = "ab" if append else "wb"
        with partial.open(mode) as output:
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())

    actual_sha256 = sha256_file(partial)
    if expected_sha256 is not None and actual_sha256.lower() != expected_sha256.lower():
        partial.unlink(missing_ok=True)
        logger.error(
            "download checksum mismatch",
            extra={"url": url, "expected_sha256": expected_sha256, "actual_sha256": actual_sha256},
        )
        raise ChecksumMismatchError(f"expected {expected_sha256}, got {actual_sha256}")

    os.replace(partial, destination)
    manifest = DumpManifest(
        url=url,
        filename=destination.name,
        sha256=actual_sha256,
        acquired_at=datetime.now(UTC).isoformat(),
        byte_size=destination.stat().st_size,
    )
    manifest_path = destination.with_suffix(destination.suffix + ".manifest.json")
    temporary_manifest = manifest_path.with_suffix(manifest_path.suffix + ".part")
    manifest_contents = json.dumps(asdict(manifest), sort_keys=True) + "\n"
    temporary_manifest.write_text(manifest_contents, encoding="utf-8")
    os.replace(temporary_manifest, manifest_path)
    duration_seconds = time.monotonic() - start
    logger.info(
        "download complete",
        extra={
            "url": url,
            "byte_size": manifest.byte_size,
            "sha256": actual_sha256,
            "duration_seconds": round(duration_seconds, 3),
            "bytes_per_second": round(manifest.byte_size / duration_seconds, 1)
            if duration_seconds > 0
            else None,
        },
    )
    return manifest
