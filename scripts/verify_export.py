"""Verification script for MARISA bundle export."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from accentuk.export import build_bundle, sha256_file
from accentuk.lexicon import Lexicon

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("verify_export")


def verify_bundle(bundle_dir: Path) -> None:
    manifest_path = bundle_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest.json missing at {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    log.info("Manifest loaded: schema_version=%s, dataset_id=%s, inventory_hash=%s",
             manifest.get("schema_version"), manifest.get("dataset_id"), manifest.get("inventory_hash"))

    # Verify checksums
    for filename, expected_hash in manifest.get("files", {}).items():
        fpath = bundle_dir / filename
        if not fpath.is_file():
            raise FileNotFoundError(f"Bundled file {filename} missing at {fpath}")
        actual_hash = sha256_file(fpath)
        if actual_hash != expected_hash:
            raise ValueError(f"Checksum mismatch for {filename}: {actual_hash} != {expected_hash}")
        log.info("Verified %s (size: %d bytes, sha256: %s...)", filename, fpath.stat().st_size, actual_hash[:8])

    # Verify lexicon lookups
    trie_path = bundle_dir / "lexicon.marisa"
    with Lexicon(trie_path) as lex:
        log.info("Lexicon opened successfully. Total entries: %d", len(lex))

        probes = [
            ("мова", True),
            ("русі", True),
            ("маю", True),
            ("замок", True),
            ("сестри", True),
            ("неіснуючеслово12345", False),
        ]
        for word, should_exist in probes:
            res = lex.lookup(word)
            if should_exist:
                if res is None:
                    raise AssertionError(f"Expected probe word '{word}' to exist in lexicon")
                log.info("Probe '%s': candidates=%s, proper=%s, readings=%d",
                         word, res["candidates"], res["proper"], len(res["readings"]))
            else:
                if res is not None:
                    raise AssertionError(f"Expected probe word '{word}' to NOT exist in lexicon")
                log.info("Probe '%s': correctly returned None", word)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export and verify accentuk MARISA bundle")
    parser.add_argument("--archive", type=Path, default=Path("data/serving"))
    parser.add_argument("--manifest", type=Path, default=Path("assets/stress-model/serving_manifest.json"))
    parser.add_argument("--assets", type=Path, default=Path("data/lexicon-assets.tar.gz"))
    parser.add_argument("--output", type=Path, default=Path("/tmp/accentuk-bundle-test"))
    args = parser.parse_args()

    log.info("Building bundle from %s to %s...", args.archive, args.output)
    build_bundle(
        archive=args.archive,
        model_manifest=args.manifest,
        output=args.output,
        assets_dir=args.assets,
    )

    log.info("Verifying bundle at %s...", args.output)
    verify_bundle(args.output)
    log.info("Bundle export and verification SUCCESSFUL!")


if __name__ == "__main__":
    main()
