"""Generate the cross-language normalization contract consumed by Go tests."""

from __future__ import annotations

import json
from pathlib import Path

from ukstress.normalizer import canonical_stressed_form, lookup_key

NORMALIZATION_VERSION = "ukstress-nfd-v1"
VECTOR_INPUTS = ("мо\u0301ва", "п'ять", "п’ять", "пів—яблука", "ҐАНОК")


def vectors() -> dict[str, object]:
    return {
        "version": NORMALIZATION_VERSION,
        "vectors": [
            {
                "input": value,
                "canonical_stressed_form": canonical_stressed_form(value),
                "lookup_key": lookup_key(value),
            }
            for value in VECTOR_INPUTS
        ],
    }


def write_vectors(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(vectors(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[3]
    write_vectors(root / "artifacts" / "normalization-conformance.json")
