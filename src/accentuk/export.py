"""Export serving archive and assets into a standalone memory-mapped MARISA bundle."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import shutil
import tarfile
import tempfile
import unicodedata
from pathlib import Path
from typing import Any

import marisa_trie

from .errors import AssetError
from .normalizer import lookup_key

DEFAULT_SUPPLEMENTARY = (5, 7, 9, 10, 12, 13, 16, 18, 19)
DEFAULT_REVIEWED = (12, 13)
DEFAULT_EXCLUSIVE = (13, 18)
SCHEMA_VERSION = 1
NORMALIZATION_VERSION = "ukstress-nfd-v1"


def drop_subsumed_free_variation(signatures: list[str]) -> list[str]:
    """Remove free-variation signatures (containing '|') when a specific member is present."""
    specific = {s for s in signatures if "|" not in s}
    if not specific:
        return signatures
    kept = []
    for s in signatures:
        subsumed = False
        if "|" in s:
            for member in s.split("|"):
                if member in specific:
                    subsumed = True
                    break
        if not subsumed:
            kept.append(s)
    return kept


def prefer_trie_default(signatures: list[str], wanted: str) -> list[str]:
    """Move wanted signature to index 0, shifting prior elements right by 1."""
    if not wanted or len(signatures) < 2 or signatures[0] == wanted:
        return signatures
    if wanted in signatures:
        idx = signatures.index(wanted)
        return [wanted] + signatures[:idx] + signatures[idx + 1 :]
    return signatures


def sha256_file(path: Path) -> str:
    """Compute hex SHA256 digest of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def validate_archive_manifest(archive_dir: Path) -> None:
    """Verify SHA256 of tables listed in MANIFEST.tsv."""
    manifest_file = archive_dir / "MANIFEST.tsv"
    if not manifest_file.is_file():
        raise AssetError(
            f"Serving archive manifest missing at {manifest_file}.",
            "Verify the archive contains a valid MANIFEST.tsv file.",
        )

    with manifest_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 4:
                continue
            table_name = parts[0]
            expected_sha = parts[3].strip()
            file_path = archive_dir / f"{table_name}.csv.gz"
            if not file_path.is_file():
                raise AssetError(
                    f"Table archive file {file_path.name} is missing.",
                    "Verify the archive was downloaded completely.",
                )
            actual_sha = sha256_file(file_path)
            if actual_sha.lower() != expected_sha.lower():
                raise AssetError(
                    f"Checksum mismatch for {file_path.name}: expected {expected_sha}, got {actual_sha}.",
                    "The archive file appears corrupted. Re-download the serving archive.",
                )


def _load_source_dictionary() -> Any:
    """Try to load the ukrainian_word_stress morphological dictionary trie."""
    try:
        from ukrainian_word_stress.stressify_ import _load_dictionary

        return _load_dictionary()
    except Exception:
        return None


def _get_readings(morph_trie: Any, word: str) -> list[list[Any]]:
    """Look up morphological readings for a word from the source dictionary trie."""
    if morph_trie is None:
        return []
    try:
        from ukrainian_word_stress.stressify_ import _parse_dictionary_value, _trie_value

        nfc_word = unicodedata.normalize("NFC", word)
        val = _trie_value(morph_trie, nfc_word)
        if val is None:
            return []
        return [(list(t), list(a)) for t, a in _parse_dictionary_value(val[0])]
    except Exception:
        return []


def _resolve_archive_path(archive: Path | str, temp_dir: Path) -> Path:
    path = Path(archive)
    if path.is_dir():
        return path
    if path.is_file() and (path.name.endswith(".tar") or path.name.endswith(".tar.gz")):
        with tarfile.open(path, "r:*") as tar:
            tar.extractall(temp_dir)
        # If files extracted directly into temp_dir or a subdirectory (like 'serving/')
        if (temp_dir / "MANIFEST.tsv").is_file():
            return temp_dir
        for sub in temp_dir.iterdir():
            if sub.is_dir() and (sub / "MANIFEST.tsv").is_file():
                return sub
        return temp_dir
    raise AssetError(f"Unsupported archive path: {path}")


def _resolve_assets_path(assets: Path | str | None, temp_dir: Path) -> Path | None:
    if assets is None:
        return None
    path = Path(assets)
    if path.is_dir():
        return path
    if path.is_file() and (path.name.endswith(".tar") or path.name.endswith(".tar.gz")):
        with tarfile.open(path, "r:*") as tar:
            tar.extractall(temp_dir)
        if (temp_dir / "suffix_table.json").is_file():
            return temp_dir
        for sub in temp_dir.iterdir():
            if sub.is_dir() and (sub / "suffix_table.json").is_file():
                return sub
        return temp_dir
    return None


def build_bundle(
    archive: Path | str,
    model_manifest: Path | str | dict[str, Any],
    output: Path | str,
    *,
    supplementary: tuple[int, ...] = DEFAULT_SUPPLEMENTARY,
    reviewed: tuple[int, ...] = DEFAULT_REVIEWED,
    exclusive: tuple[int, ...] = DEFAULT_EXCLUSIVE,
    assets_dir: Path | str | None = None,
    include_all_readings: bool = False,
    validate_manifest: bool = True,
) -> Path:
    """Build a versioned MARISA lexicon bundle from a serving archive."""
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as temp_str:
        temp_dir = Path(temp_str)
        archive_dir = _resolve_archive_path(archive, temp_dir / "archive")
        resolved_assets_dir = _resolve_assets_path(assets_dir, temp_dir / "assets")

        if validate_manifest:
            validate_archive_manifest(archive_dir)

        # Parse model manifest to get inventory_hash
        if isinstance(model_manifest, (str, Path)):
            manifest_content = Path(model_manifest).read_text(encoding="utf-8")
            manifest_data = json.loads(manifest_content)
        else:
            manifest_data = dict(model_manifest)
        inventory_hash = str(manifest_data.get("inventory_hash", ""))

        # 1. Read active dataset ID
        active_dataset_file = archive_dir / "active_dataset.csv.gz"
        active_dataset_id = 2
        if active_dataset_file.is_file():
            with gzip.open(active_dataset_file, "rt", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    active_dataset_id = int(row["dataset_id"])
                    break

        allowed_datasets = {active_dataset_id, *supplementary}
        reviewed_set = set(reviewed)
        exclusive_set = set(exclusive)

        # 2. Read trie defaults map if present
        trie_defaults: dict[str, str] = {}
        trie_defaults_file = None
        if resolved_assets_dir and (resolved_assets_dir / "trie_defaults_map.json").is_file():
            trie_defaults_file = resolved_assets_dir / "trie_defaults_map.json"
        elif (archive_dir / "trie_defaults_map.json").is_file():
            trie_defaults_file = archive_dir / "trie_defaults_map.json"

        if trie_defaults_file and trie_defaults_file.is_file():
            trie_defaults = json.loads(trie_defaults_file.read_text(encoding="utf-8"))

        # 3. Stream and aggregate candidates
        # word -> sig -> [all_upper: bool, any_reviewed: bool, any_exclusive: bool, max_conf: float, min_rank: int]
        aggregated: dict[str, dict[str, list[Any]]] = {}

        # 3a. stress_lookup.csv.gz
        stress_lookup_file = archive_dir / "stress_lookup.csv.gz"
        if not stress_lookup_file.is_file():
            raise AssetError(f"Missing {stress_lookup_file}")

        with gzip.open(stress_lookup_file, "rt", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            # Find column indices
            ds_col = header.index("dataset_id")
            word_col = header.index("form_normalized")
            stressed_col = header.index("stressed_form")
            sig_col = header.index("stress_signature")
            conf_col = header.index("confidence")
            rank_col = header.index("source_rank")

            for row in reader:
                ds_id = int(row[ds_col])
                if ds_id not in allowed_datasets:
                    continue
                word = lookup_key(row[word_col])
                stressed_form = row[stressed_col]
                sig = row[sig_col]
                conf = float(row[conf_col])
                rank = int(row[rank_col])
                is_up = bool(stressed_form and stressed_form[0].isupper())
                is_rev = ds_id in reviewed_set
                is_excl = ds_id in exclusive_set

                word_entry = aggregated.get(word)
                if word_entry is None:
                    aggregated[word] = {sig: [is_up, is_rev, is_excl, conf, rank]}
                else:
                    sig_entry = word_entry.get(sig)
                    if sig_entry is None:
                        word_entry[sig] = [is_up, is_rev, is_excl, conf, rank]
                    else:
                        if not is_up:
                            sig_entry[0] = False
                        if is_rev:
                            sig_entry[1] = True
                        if is_excl:
                            sig_entry[2] = True
                        if conf > sig_entry[3]:
                            sig_entry[3] = conf
                        if rank < sig_entry[4]:
                            sig_entry[4] = rank

        # 3b. contextual_stress_candidate.csv.gz
        contextual_file = archive_dir / "contextual_stress_candidate.csv.gz"
        if contextual_file.is_file():
            with gzip.open(contextual_file, "rt", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader)
                hash_col = header.index("inventory_hash")
                word_col = header.index("form_normalized")
                sig_col = header.index("stress_signature")
                stressed_col = header.index("stressed_form")

                for row in reader:
                    if row[hash_col] != inventory_hash:
                        continue
                    word = lookup_key(row[word_col])
                    stressed_form = row[stressed_col]
                    sig = row[sig_col]
                    conf = 0.0
                    rank = 2147483647
                    is_up = bool(stressed_form and stressed_form[0].isupper())
                    is_rev = False
                    is_excl = False

                    word_entry = aggregated.get(word)
                    if word_entry is None:
                        aggregated[word] = {sig: [is_up, is_rev, is_excl, conf, rank]}
                    else:
                        sig_entry = word_entry.get(sig)
                        if sig_entry is None:
                            word_entry[sig] = [is_up, is_rev, is_excl, conf, rank]
                        else:
                            if not is_up:
                                sig_entry[0] = False
                            if conf > sig_entry[3]:
                                sig_entry[3] = conf
                            if rank < sig_entry[4]:
                                sig_entry[4] = rank

        # 4. Load morphological dictionary. Without it every entry gets empty readings and the
        # morphology tier silently decides nothing (it cost 5.5 points of lang-uk heteronyms), so a
        # missing ukrainian-word-stress is an error, not a fallback.
        morph_trie = _load_source_dictionary()
        if morph_trie is None:
            raise RuntimeError(
                "ukrainian-word-stress is required to build a bundle (the morphology tier reads its "
                "readings): pip install 'accentuk[build]'"
            )

        # 5. Build final entries for BytesTrie
        trie_entries: list[tuple[str, bytes]] = []

        for word, sig_dict in aggregated.items():
            # Sort signatures by:
            # min(is_upper) ASC (0 before 1), max_conf DESC, min_rank ASC, sig ASC
            sorted_sigs = sorted(
                sig_dict.keys(),
                key=lambda s: (
                    1 if sig_dict[s][0] else 0,
                    -sig_dict[s][3],
                    sig_dict[s][4],
                    s,
                ),
            )

            # Record first reviewed / exclusive in query order
            decided = None
            only = None
            proper_map: dict[str, bool] = {}

            for s in sorted_sigs:
                all_upper, any_rev, any_excl, _, _ = sig_dict[s]
                # In SQL: bool_and(stressed_form ~ '^[[:upper:]]') AND bool_or(reviewed)
                proper_map[s] = bool(all_upper and any_rev)
                if any_rev and decided is None:
                    decided = s
                if any_excl and only is None:
                    only = s

            # Postprocess signatures
            candidates = drop_subsumed_free_variation(sorted_sigs)
            if only is not None:
                candidates = [only]
            elif decided is not None:
                candidates = prefer_trie_default(candidates, decided)
            else:
                candidates = prefer_trie_default(candidates, trie_defaults.get(word, ""))

            # Fetch morphological readings
            readings: list[list[Any]] = []
            if include_all_readings or len(candidates) > 1:
                readings = _get_readings(morph_trie, word)

            record = {
                "candidates": candidates,
                "proper": proper_map,
                "readings": readings,
            }
            payload = json.dumps(record, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
            trie_entries.append((word, payload))

        # Sort entries by key for deterministic BytesTrie construction
        trie_entries.sort(key=lambda x: x[0])
        trie = marisa_trie.BytesTrie(trie_entries)
        trie_path = output_path / "lexicon.marisa"
        trie.save(str(trie_path))

        # 6. Copy small asset files if available
        files_to_hash = ["lexicon.marisa"]
        for asset_name in ("suffix_table.json", "counted_forms.json", "trie_defaults_map.json"):
            src_file = None
            if resolved_assets_dir and (resolved_assets_dir / asset_name).is_file():
                src_file = resolved_assets_dir / asset_name
            elif (archive_dir / asset_name).is_file():
                src_file = archive_dir / asset_name

            dest_file = output_path / asset_name
            if src_file and src_file.is_file():
                shutil.copy2(src_file, dest_file)
            elif not dest_file.is_file():
                # Write empty JSON dict/list as fallback placeholder
                dest_file.write_text("{}" if asset_name != "counted_forms.json" else "[]", encoding="utf-8")
            files_to_hash.append(asset_name)

        # 7. Write bundle manifest
        file_hashes = {name: sha256_file(output_path / name) for name in sorted(files_to_hash)}
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "normalization_version": NORMALIZATION_VERSION,
            "dataset_id": active_dataset_id,
            "inventory_hash": inventory_hash,
            "model_version": str(manifest_data.get("model_version", "v3-xenc")),
            "files": file_hashes,
        }
        (output_path / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return output_path
