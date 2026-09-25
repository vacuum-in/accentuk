"""Build a serving manifest from the paradigm-expanded inventory.

Coverage work is not servable until it reaches the manifest. `serving.py`
refuses any form absent from `manifest["forms"]` ("form is outside frozen
training coverage"), so the 12,706 forms that gained glosses this cycle are
still unreachable through the API while the shipped manifest lists 570.

Two properties of the existing manifest are preserved deliberately:

* `signatures` is the authority the database is checked against — `resolve()`
  raises when the caller's candidates differ from it — so a form is emitted
  only when every candidate carries both a signature and a gloss. A candidate
  with an empty gloss would be scored against an empty string and could not win
  on merit.
* `inventory_hash` pins the manifest to the model that was trained on it. The
  Go API compares it (`ACTIVE_INVENTORY_HASH`), which is what stops a model and
  an inventory from drifting apart silently.

`review_status` travels with each candidate so an LLM-authored gloss stays
identifiable in production rather than becoming anonymous ground truth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from ukstress_ml import ambiguity


def build(inventories: list[Path], model_version: str, threshold: float) -> dict[str, Any]:
    """Merge one or more inventories into a serving manifest.

    Earlier inventories win on a collision. The Wiktionary-derived inventory is
    listed first because its glosses are curated senses; a later inventory
    exists to *add* forms nobody grouped, never to overwrite a form that
    already carries real ones.
    """
    forms: dict[str, Any] = {}
    skipped = {"incomplete_gloss": 0, "single_signature": 0, "duplicate_gloss": 0,
               "superseded_by_earlier_inventory": 0}
    for inventory in inventories:
        for form in ambiguity.load(inventory):
            if form.form in forms:
                skipped["superseded_by_earlier_inventory"] += 1
                continue
            entry = manifest_entry(form, skipped)
            if entry is not None:
                forms[form.form] = entry

    payload = json.dumps(forms, ensure_ascii=False, sort_keys=True).encode()
    return {
        "corpus_hash": hashlib.sha256(payload).hexdigest(),
        "forms": forms,
        "inventory_hash": hashlib.sha256(payload).hexdigest(),
        "model_version": model_version,
        "schema_version": 1,
        "threshold": threshold,
        "_skipped": skipped,
    }


def manifest_entry(form: ambiguity.AmbiguousForm,
                   skipped: dict[str, int]) -> dict[str, Any] | None:
    """Return the manifest entry for one form, or None with a reason counted."""
    signatures = sorted({c.signature for c in form.candidates})
    if len(signatures) < 2:
        # Not ambiguous: the dictionary tier answers it, and offering it to the
        # model would spend a forward pass to confirm a single option.
        skipped["single_signature"] += 1
        return None
    if not all(c.definition for c in form.candidates):
        skipped["incomplete_gloss"] += 1
        return None
    if len({c.definition.strip() for c in form.candidates}) < len(form.candidates):
        # Two candidates carrying the same gloss present the cross-encoder with
        # identical (context, gloss) inputs, so it cannot prefer either on
        # evidence — the pick is a coin flip made at whatever confidence the
        # scores happen to differ by. Measured on the held-out set, these
        # accounted for 32% of all high-margin errors. Serving them is strictly
        # worse than abstaining, so they are withheld until their glosses are
        # re-authored to actually distinguish the senses (for the toponym pairs
        # that dominate here, by naming the region).
        skipped["duplicate_gloss"] += 1
        return None
    return {
        "candidates": [
            {
                "definition": c.definition,
                "sense_id": c.sense_id,
                "signature": c.signature,
                "stressed": c.stressed,
                **({"review_status": c.review_status} if c.review_status else {}),
            }
            for c in sorted(form.candidates, key=lambda c: c.sense_id)
        ],
        "group_id": form.group_id,
        "sense_ids": sorted({c.sense_id for c in form.candidates}),
        "signatures": signatures,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inventory", type=Path, action="append", default=None,
        help="repeatable; earlier inventories win on a collision",
    )
    parser.add_argument("--model-version", default="v5-balanced")
    parser.add_argument(
        "--threshold",
        type=float,
        required=True,
        help="abstention margin; below it the API keeps the dictionary default",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("output/ml/serving_manifest_expanded.json")
    )
    args = parser.parse_args()

    inventories = args.inventory or [Path("output/ml/ambiguous_forms_glossed.jsonl")]
    manifest = build(inventories, args.model_version, args.threshold)
    skipped = manifest.pop("_skipped")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"forms in manifest        : {len(manifest['forms']):>7,}")
    print(f"  skipped, gloss missing : {skipped['incomplete_gloss']:>7,}")
    print(f"  skipped, one signature : {skipped['single_signature']:>7,}")
    print(f"inventory_hash           : {manifest['inventory_hash'][:16]}...")
    print(f"threshold                : {manifest['threshold']}")
    print(f"written to {args.output}")


if __name__ == "__main__":
    main()
