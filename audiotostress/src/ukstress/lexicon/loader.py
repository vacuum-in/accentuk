"""Load versioned stress dictionaries from external data files."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

from ukstress.config.models import LexiconConfig
from ukstress.datasets import StressCandidate
from ukstress.lexicon.stress import apply_stress, normalize_surface, parse_stress
from ukstress.provenance.fingerprints import content_fingerprint, file_fingerprint


class InMemoryStressLexicon:
    """Small immutable lexicon useful for tests and programmatic pipeline runs."""

    def __init__(
        self,
        entries: Mapping[str, Sequence[StressCandidate]],
        *,
        version: str,
        fingerprint: str,
        provenance: Mapping[str, str] | None = None,
    ) -> None:
        if not version.strip() or not fingerprint.strip():
            raise ValueError("version and fingerprint must be non-empty")
        self._version = version
        self._fingerprint = fingerprint
        self._provenance = MappingProxyType(dict(provenance or {"source": "memory"}))
        self._entries = MappingProxyType(
            {
                normalize_surface(surface): tuple(candidates)
                for surface, candidates in entries.items()
            }
        )

    @property
    def version(self) -> str:
        return self._version

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    @property
    def provenance(self) -> Mapping[str, str]:
        return self._provenance

    def lookup(self, surface: str) -> tuple[StressCandidate, ...]:
        return self._entries.get(normalize_surface(surface), ())


class FileStressLexicon:
    """Immutable, content-addressed lexicon loaded from an external file."""

    def __init__(
        self,
        *,
        version: str,
        entries: Mapping[str, tuple[StressCandidate, ...]],
        fingerprint: str,
        provenance: Mapping[str, str],
    ) -> None:
        self._version = version
        self._entries = MappingProxyType(dict(entries))
        self._fingerprint = fingerprint
        self._provenance = MappingProxyType(dict(provenance))

    @property
    def version(self) -> str:
        return self._version

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    @property
    def provenance(self) -> Mapping[str, str]:
        return self._provenance

    def lookup(self, surface: str) -> tuple[StressCandidate, ...]:
        return self._entries.get(normalize_surface(surface), ())

    @classmethod
    def from_config(cls, config: LexiconConfig) -> FileStressLexicon:
        return cls.load(config.path, version=config.version, source_name=config.source_name)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        version: str,
        source_name: str | None = None,
    ) -> FileStressLexicon:
        source_path = Path(path)
        if not version.strip():
            raise ValueError("lexicon version must be non-empty")
        raw_entries, embedded_version = _load_raw_entries(source_path)
        if embedded_version is not None and embedded_version != version:
            raise ValueError(
                f"configured lexicon version {version!r} does not match file version "
                f"{embedded_version!r}"
            )
        default_source = source_name or source_path.stem
        grouped: dict[str, dict[tuple[str, int], set[str]]] = defaultdict(lambda: defaultdict(set))
        for row_number, raw_entry in enumerate(raw_entries, start=1):
            try:
                surface, stressed_form, vowel_index, source = _parse_entry(
                    raw_entry, default_source=default_source
                )
            except (TypeError, ValueError) as error:
                raise ValueError(f"invalid lexicon entry {row_number}: {error}") from error
            grouped[surface][(stressed_form, vowel_index)].add(source)

        entries: dict[str, tuple[StressCandidate, ...]] = {}
        for surface, variants in grouped.items():
            entries[surface] = tuple(
                StressCandidate(
                    stressed_form=stressed_form,
                    vowel_index=vowel_index,
                    source=";".join(sorted(sources)),
                )
                for (stressed_form, vowel_index), sources in sorted(
                    variants.items(), key=lambda item: (item[0][1], item[0][0])
                )
            )

        source_fingerprint = file_fingerprint(source_path, namespace="lexicon-source")
        semantic_entries = {
            surface: [candidate.model_dump(mode="json") for candidate in candidates]
            for surface, candidates in sorted(entries.items())
        }
        fingerprint = content_fingerprint(
            {
                "version": version,
                "source_fingerprint": source_fingerprint,
                "entries": semantic_entries,
            },
            namespace="stress-lexicon",
        )
        provenance = {
            "path": str(source_path),
            "version": version,
            "source_name": default_source,
            "source_fingerprint": source_fingerprint,
            "format": source_path.suffix.casefold().removeprefix("."),
        }
        return cls(
            version=version,
            entries=entries,
            fingerprint=fingerprint,
            provenance=provenance,
        )


def _load_raw_entries(path: Path) -> tuple[list[Mapping[str, Any]], str | None]:
    suffix = path.suffix.casefold()
    if suffix == ".jsonl":
        entries: list[Mapping[str, Any]] = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON on line {line_number}: {error.msg}") from error
            if not isinstance(raw, dict):
                raise ValueError(f"JSONL line {line_number} must be an object")
            entries.append(raw)
        return entries, None
    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        with path.open(encoding="utf-8", newline="") as stream:
            return list(csv.DictReader(stream, delimiter=delimiter)), None
    if suffix in {".json", ".yaml", ".yml"}:
        text = path.read_text(encoding="utf-8")
        raw_document = json.loads(text) if suffix == ".json" else yaml.safe_load(text)
        return _entries_from_document(raw_document)
    raise ValueError(f"unsupported lexicon format {suffix!r}")


def _entries_from_document(document: Any) -> tuple[list[Mapping[str, Any]], str | None]:
    embedded_version: str | None = None
    if isinstance(document, dict):
        raw_version = document.get("version")
        if raw_version is not None:
            embedded_version = str(raw_version)
        document = document.get("entries")
    if not isinstance(document, list) or not all(isinstance(item, dict) for item in document):
        raise ValueError("lexicon document must be a list or a mapping containing an entries list")
    return document, embedded_version


def _parse_entry(raw: Mapping[str, Any], *, default_source: str) -> tuple[str, str, int, str]:
    stressed_raw = raw.get("stressed_form")
    surface_raw = raw.get("surface")
    index_raw = raw.get("vowel_index")
    source = str(raw.get("source") or default_source).strip()
    if not source:
        raise ValueError("source must be non-empty")

    if stressed_raw is not None and str(stressed_raw).strip():
        parsed = parse_stress(str(stressed_raw))
        surface = parsed.surface
        stressed_form = parsed.stressed_form
        vowel_index = parsed.vowel_index
        if surface_raw is not None and normalize_surface(str(surface_raw)) != surface:
            raise ValueError("surface does not match stressed_form")
        if index_raw not in (None, "") and _parse_index(index_raw) != vowel_index:
            raise ValueError("vowel_index does not match stressed_form")
    else:
        if surface_raw is None or index_raw in (None, ""):
            raise ValueError("entry needs stressed_form or both surface and vowel_index")
        surface = normalize_surface(str(surface_raw))
        vowel_index = _parse_index(index_raw)
        stressed_form = apply_stress(surface, vowel_index)
    return surface, stressed_form, vowel_index, source


def _parse_index(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("vowel_index must be an integer")
    try:
        index = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError("vowel_index must be an integer") from error
    if str(index) != str(value).strip() and not isinstance(value, int):
        raise ValueError("vowel_index must be an integer")
    if index < 0:
        raise ValueError("vowel_index must be non-negative")
    return index
