"""License filtering, PII minimization, and metadata safety checks."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlparse

from ukstress.datasets import CorpusManifestRecord, MiningRecord

_SAFE_METADATA = re.compile(r"^[^\x00\r\n]+$")


class ExportLicensePolicy:
    """Configurable license policy for derived exports and optional media references."""

    def __init__(
        self,
        *,
        allowed_license_ids: frozenset[str] | None = None,
        require_commercial_use: bool = False,
        require_redistribution: bool = False,
    ) -> None:
        self.allowed_license_ids = allowed_license_ids
        self.require_commercial_use = require_commercial_use
        self.require_redistribution = require_redistribution

    def signature(self) -> dict[str, object]:
        return {
            "allowed_license_ids": sorted(self.allowed_license_ids)
            if self.allowed_license_ids is not None
            else None,
            "require_commercial_use": self.require_commercial_use,
            "require_redistribution": self.require_redistribution,
        }

    def allows(self, record: CorpusManifestRecord | MiningRecord) -> bool:
        if (
            self.allowed_license_ids is not None
            and record.license_id not in self.allowed_license_ids
        ):
            return False
        if self.require_commercial_use and record.commercial_use is not True:
            return False
        return not (self.require_redistribution and record.redistribution is not True)


def redact_speaker_id(speaker_id: str | None) -> None:
    """Return no speaker identifier for public exports; internal records retain it for splits."""

    del speaker_id
    return None


def validate_media_uri(uri: str, *, base_dir: str | Path | None = None) -> str:
    """Validate local/file media references without shell interpolation or traversal."""

    if not uri or not _SAFE_METADATA.fullmatch(uri):
        raise ValueError("media URI must be non-empty and contain no control characters")
    parsed = urlparse(uri)
    if parsed.scheme not in {"", "file"}:
        raise ValueError("only local paths and file:// media URIs are supported")
    if parsed.scheme == "file" and parsed.netloc not in {"", "localhost"}:
        raise ValueError("remote file authorities are forbidden")
    path = Path(parsed.path if parsed.scheme == "file" else uri)
    if base_dir is not None and not path.is_absolute():
        path = Path(base_dir) / path
    if any(part == ".." for part in path.parts):
        raise ValueError("media path traversal is forbidden")
    return str(path)


def validate_metadata_text(value: str, *, field: str = "metadata") -> str:
    """Reject control characters that could become shell/log/header injection."""

    if not isinstance(value, str) or not _SAFE_METADATA.fullmatch(value):
        raise ValueError(f"{field} contains unsafe control characters")
    return value
