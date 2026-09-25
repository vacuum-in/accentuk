#!/usr/bin/env python3
"""Build the immutable runtime coverage manifest from the actual train split.

This intentionally reads the frozen inventory, never homographs.db.  The
SQLite source belongs to the offline inventory-building workflow only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--release-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    release = json.loads(args.release_report.read_text(encoding="utf-8"))
    if not release.get("release_eligible"):
        raise SystemExit("refusing to publish a model that did not pass release gates")

    represented: dict[str, dict[str, object]] = {}
    with args.train.open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            form = row["form"]
            entry = represented.setdefault(
                form,
                {"group_id": row["group_id"], "sense_ids": set(), "signatures": set()},
            )
            entry["sense_ids"].update(  # type: ignore[union-attr]
                candidate["sense_id"] for candidate in row["candidates"]
            )
            entry["signatures"].update(  # type: ignore[union-attr]
                candidate["signature"] for candidate in row["candidates"]
            )

    inventory: dict[str, dict[str, str]] = {}
    with args.inventory.open(encoding="utf-8") as source:
        for line in source:
            sense = json.loads(line)
            inventory[sense["sense_id"]] = {
                "signature": sense["signature"],
                "stressed": sense["stressed"],
                "definition": sense["definition"],
            }

    forms: dict[str, object] = {}
    for form, raw in sorted(represented.items()):
        sense_ids = sorted(raw["sense_ids"])  # type: ignore[arg-type]
        candidates = []
        for sense_id in sense_ids:
            sense = inventory[sense_id]
            candidates.append({"sense_id": sense_id, **sense})
        # A trained form's allowed set is the complete candidate set observed
        # in training rows, including a sense that happened not to be gold.
        signatures = sorted(raw["signatures"])  # type: ignore[arg-type]
        forms[form] = {
            "group_id": raw["group_id"],
            "sense_ids": sense_ids,
            "signatures": signatures,
            "candidates": candidates,
        }

    train_hash = hashlib.sha256(args.train.read_bytes()).hexdigest()
    payload = {
        "schema_version": 1,
        "model_version": args.release_report.parent.name,
        "inventory_hash": release["inventory_hash"],
        "corpus_hash": train_hash,
        "threshold": release["abstention"]["threshold"],
        "forms": forms,
    }
    encoded = (json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(encoded)
    print(hashlib.sha256(encoded).hexdigest())


if __name__ == "__main__":
    main()
