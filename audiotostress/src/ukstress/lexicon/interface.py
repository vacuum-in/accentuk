"""Backend-neutral stress lexicon interface."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from ukstress.datasets import StressCandidate


class StressLexicon(Protocol):
    """A versioned lookup that never resolves ambiguous variants implicitly."""

    @property
    def version(self) -> str: ...

    @property
    def fingerprint(self) -> str: ...

    @property
    def provenance(self) -> Mapping[str, str]: ...

    def lookup(self, surface: str) -> tuple[StressCandidate, ...]:
        """Return every valid candidate for a normalized surface, or an empty tuple."""
        ...
