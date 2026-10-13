"""Exception classes for accentuk."""

from __future__ import annotations


class AccentukError(Exception):
    """Base exception for accentuk errors."""


class AssetError(AccentukError):
    """Raised when an asset bundle or file is missing, incompatible, or corrupted.

    Includes actionable repair instructions.
    """

    def __init__(self, message: str, repair: str | None = None) -> None:
        self.message = message
        self.repair = repair or "Run 'accentuk download' to fetch or repair the required assets."
        super().__init__(f"{self.message}\nResolution: {self.repair}")


class NormalizationError(AccentukError, ValueError):
    """Raised when a word or signature cannot be normalized or validated."""


class ServingError(AccentukError, ValueError):
    """Raised when the inference contract or target validation is violated."""
