"""Memory-mapped MARISA lexicon reader."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import marisa_trie

from .errors import AssetError
from .normalizer import lookup_key


class Lexicon:
    """Memory-mapped Ukrainian stress lexicon backed by marisa_trie.BytesTrie."""

    SCHEMA_VERSION = 1
    NORMALIZATION_VERSION = "ukstress-nfd-v1"

    def __init__(self, trie_path: Path | str) -> None:
        path = Path(trie_path)
        if not path.is_file():
            raise AssetError(
                f"Lexicon trie file not found at {path}.",
                "Run 'accentuk download' or 'accentuk export' to generate the lexicon.",
            )
        self.path = path
        self._trie: marisa_trie.BytesTrie | None = marisa_trie.BytesTrie()
        try:
            self._trie.mmap(str(path))
        except Exception as err:
            raise AssetError(
                f"Failed to memory-map lexicon trie at {path}: {err}",
                "The lexicon file may be corrupted. Run 'accentuk download' to re-download it.",
            ) from err
        self._closed = False

    def lookup(self, word: str, *, raw_key: bool = False) -> dict[str, Any] | None:
        """Look up a word in the memory-mapped lexicon.

        Returns decoded payload dict with keys:
          - candidates: list[str]
          - proper: dict[str, bool]
          - readings: list[list] (tag list plus character accent offsets)
        or None if not found.
        """
        if self._closed or self._trie is None:
            raise RuntimeError("Lexicon is closed.")

        key = word if raw_key else lookup_key(word)
        payloads = self._trie.get(key)
        if payloads is None or len(payloads) == 0:
            return None
        if len(payloads) > 1:
            raise AssetError(
                f"Corrupt lexicon: multiple payloads ({len(payloads)}) found for key '{key}'.",
                "Rebuild or re-download the lexicon bundle.",
            )

        raw = payloads[0]
        try:
            data = json.loads(raw.decode("utf-8"))
        except Exception as err:
            raise AssetError(
                f"Malformed JSON payload for key '{key}': {err}",
                "Rebuild or re-download the lexicon bundle.",
            ) from err

        if not isinstance(data, dict):
            raise AssetError(
                f"Malformed payload schema for key '{key}': expected dict, got {type(data).__name__}.",
                "Rebuild or re-download the lexicon bundle.",
            )

        if "candidates" not in data or not isinstance(data["candidates"], list):
            raise AssetError(
                f"Malformed payload schema for key '{key}': missing or invalid 'candidates'.",
                "Rebuild or re-download the lexicon bundle.",
            )

        if "proper" not in data or not isinstance(data["proper"], dict):
            raise AssetError(
                f"Malformed payload schema for key '{key}': missing or invalid 'proper'.",
                "Rebuild or re-download the lexicon bundle.",
            )

        if "readings" not in data or not isinstance(data["readings"], list):
            raise AssetError(
                f"Malformed payload schema for key '{key}': missing or invalid 'readings'.",
                "Rebuild or re-download the lexicon bundle.",
            )

        return data

    def close(self) -> None:
        """Close the memory-mapped trie."""
        self._closed = True
        self._trie = None

    def __enter__(self) -> Lexicon:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def __len__(self) -> int:
        if self._closed or self._trie is None:
            return 0
        return len(self._trie)
