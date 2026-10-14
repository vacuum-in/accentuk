"""Asset bundle verification, atomic download, and cache management."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import urllib.request
from pathlib import Path
from typing import Any

from .errors import AssetError
from .export import build_bundle, sha256_file

log = logging.getLogger("accentuk.assets")

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "accentuk"


def get_releases_descriptor() -> dict[str, Any]:
    """Load bundled releases.json descriptor."""
    desc_file = Path(__file__).parent / "assets" / "releases.json"
    if not desc_file.is_file():
        raise AssetError("releases.json descriptor is missing.")
    return json.loads(desc_file.read_text(encoding="utf-8"))


class AssetBundle:
    """Represents an active, verified accentuk asset bundle."""

    def __init__(self, path: Path, manifest: dict[str, Any]) -> None:
        self.path = path
        self.manifest = manifest
        self.schema_version = manifest.get("schema_version", 1)
        self.normalization_version = manifest.get("normalization_version", "ukstress-nfd-v1")
        self.dataset_id = manifest.get("dataset_id", 2)
        self.inventory_hash = manifest.get("inventory_hash", "")
        self.model_version = manifest.get("model_version", "")

    @property
    def lexicon_path(self) -> Path:
        return self.path / "lexicon.marisa"

    @classmethod
    def open(cls, directory: Path | str) -> AssetBundle:
        """Open and validate an existing asset bundle directory."""
        dir_path = Path(directory)
        manifest_file = dir_path / "manifest.json"
        if not manifest_file.is_file():
            raise AssetError(
                f"Asset bundle manifest missing at {manifest_file}.",
                "Run 'accentuk download' to fetch or repair the bundle.",
            )

        try:
            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        except Exception as err:
            raise AssetError(
                f"Corrupt bundle manifest at {manifest_file}: {err}",
                "Run 'accentuk download' to re-download the bundle.",
            ) from err

        if manifest.get("schema_version") != 1:
            raise AssetError(
                f"Incompatible bundle schema version: {manifest.get('schema_version')}, expected 1.",
                "Upgrade accentuk or download a compatible bundle.",
            )
        if manifest.get("normalization_version") != "ukstress-nfd-v1":
            raise AssetError(
                f"Incompatible normalization version: {manifest.get('normalization_version')}, expected 'ukstress-nfd-v1'.",
                "Rebuild or re-download the bundle.",
            )

        # Verify bundled file checksums
        files = manifest.get("files", {})
        for filename, expected_sha in files.items():
            fpath = dir_path / filename
            if not fpath.is_file():
                raise AssetError(
                    f"Required bundle file missing: {fpath}.",
                    "Run 'accentuk download' to repair the bundle.",
                )
            actual_sha = sha256_file(fpath)
            if actual_sha.lower() != expected_sha.lower():
                raise AssetError(
                    f"Corrupted bundle file {filename}: expected {expected_sha}, got {actual_sha}.",
                    "Run 'accentuk download' to re-download and repair.",
                )

        return cls(dir_path, manifest)


def _download_file(url: str, dest: Path, expected_sha: str | None = None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp_dest = dest.with_suffix(dest.suffix + ".part")
    from . import __version__

    try:
        req = urllib.request.Request(url, headers={"User-Agent": f"accentuk/{__version__}"})
        with urllib.request.urlopen(req) as resp, temp_dest.open("wb") as out:
            shutil.copyfileobj(resp, out)

        if expected_sha:
            actual = sha256_file(temp_dest)
            if actual.lower() != expected_sha.lower():
                raise AssetError(
                    f"Checksum mismatch for {dest.name}: expected {expected_sha}, got {actual}."
                )
        temp_dest.replace(dest)
    finally:
        if temp_dest.is_file():
            temp_dest.unlink(missing_ok=True)


def download_assets(
    target_dir: Path | str | None = None,
    *,
    include_tts: bool = False,
    allow_local_fixtures: bool = True,
    light: bool = False,
    full: bool = False,
) -> Path:
    """Atomically download or build and verify the stress assets bundle."""
    target_path = Path(target_dir) if target_dir else DEFAULT_CACHE_DIR
    target_path.mkdir(parents=True, exist_ok=True)

    desc = get_releases_descriptor()
    staging_dir = target_path.parent / f".staging-{os.getpid()}-{target_path.name}"
    if staging_dir.is_dir():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Optional local sources instead of downloads (maintainers): ACCENTUK_LOCAL_DATA holds serving/ and
        # lexicon-assets.tar.gz, ACCENTUK_LOCAL_MODELS the model directories, ACCENTUK_LOCAL_CT2 the CT2 ones.
        local_data = Path(os.environ.get("ACCENTUK_LOCAL_DATA", "/nonexistent"))
        local_archive = local_data / "serving"
        local_assets_tar = local_data / "lexicon-assets.tar.gz"
        local_models = Path(os.environ.get("ACCENTUK_LOCAL_MODELS", "/nonexistent"))

        if allow_local_fixtures and local_archive.is_dir() and (local_models / "stress-model").is_dir():
            log.info("Using local archive fixtures from %s...", local_archive)
            build_bundle(
                archive=local_archive,
                model_manifest=local_models / "stress-model" / "serving_manifest.json",
                output=staging_dir,
                assets_dir=local_assets_tar if local_assets_tar.is_file() else local_models / "lexicon",
            )
            # Copy CT2 light models if present
            ct2_source = Path(os.environ.get("ACCENTUK_LOCAL_CT2", "/nonexistent"))
            if ct2_source.is_dir():
                for ct2_name in ("stress-model-ct2-int8", "tok-v5-ct2-int8", "verbalizer-ct2-int8", "gemma-verbalizer-ct2-int8", "verbalizer-spans-ct2-int8"):
                    src_ct2 = ct2_source / ct2_name
                    if src_ct2.is_dir():
                        shutil.copytree(src_ct2, staging_dir / ct2_name, dirs_exist_ok=True)

            # Copy full models unless light-only was explicitly requested
            if not light or full:
                for model_name in ("stress-model", "tok-v5", "combiner-v1", "spacy-uk_core_news_trf_380"):
                    src_m = local_models / model_name
                    if src_m.is_dir():
                        shutil.copytree(src_m, staging_dir / model_name, dirs_exist_ok=True)

                if include_tts:
                    src_verb = local_models / "verbalizer"
                    if src_verb.is_dir():
                        shutil.copytree(src_verb, staging_dir / "verbalizer", dirs_exist_ok=True)
        else:
            # The lexicon comes prebuilt (building it needs ukrainian-word-stress for the readings),
            # the models from the Hub at the commits pinned in releases.json.
            lex = desc["lexicon_bundle"]
            downloads_dir = staging_dir / "downloads"
            downloads_dir.mkdir(exist_ok=True)
            lex_tar = downloads_dir / "lexicon.tar.gz"
            _download_file(lex["url"], lex_tar, lex["sha256"])
            import tarfile
            with tarfile.open(lex_tar) as tf:
                tf.extractall(staging_dir, filter="data")
            shutil.rmtree(downloads_dir)

            from huggingface_hub import snapshot_download
            sets = {"full"} if full and not light else {"light"}
            if include_tts:
                sets |= {f"{x}-tts" for x in sets}
            for name, m in desc["hub_models"].items():
                if not sets & set(m["set"] if isinstance(m["set"], list) else [m["set"]]):
                    continue
                log.info("Downloading %s (%s@%s)...", name, m["repo"], m["revision"][:12])
                snapshot_download(m["repo"], revision=m["revision"], local_dir=staging_dir / m["target"],
                                  ignore_patterns=["*.whl", ".gitattributes", "README.md"])
            # Both cross-encoders read the serving manifest from the bundle root.
            for cand in ("ct2_models/stress-model-ct2-int8", "stress-model"):
                if (staging_dir / cand / "serving_manifest.json").is_file():
                    shutil.copy(staging_dir / cand / "serving_manifest.json", staging_dir / "serving_manifest.json")
                    break

        # Validate staging before activating
        AssetBundle.open(staging_dir)

        # Replace what was staged; directories nested under ct2_models/ are replaced one by one, so adding
        # --tts to an installed bundle keeps its other CT2 models.
        for item in staging_dir.iterdir():
            dest = target_path / item.name
            if item.is_dir() and item.name == "ct2_models":
                dest.mkdir(exist_ok=True)
                for sub in item.iterdir():
                    if (dest / sub.name).is_dir():
                        shutil.rmtree(dest / sub.name)
                    shutil.copytree(sub, dest / sub.name)
            elif item.is_dir():
                if dest.is_dir():
                    shutil.rmtree(dest)
                shutil.copytree(item, dest)
            else:
                shutil.copy2(item, dest)

        log.info("Assets successfully installed and verified at %s", target_path)
        return target_path
    finally:
        if staging_dir.is_dir():
            shutil.rmtree(staging_dir, ignore_errors=True)
