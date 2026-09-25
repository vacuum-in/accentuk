import json
from pathlib import Path

from ukstress.conformance import vectors


def test_checked_in_conformance_vectors_are_generated() -> None:
    artifact = Path(__file__).resolve().parents[2] / "artifacts" / "normalization-conformance.json"
    assert json.loads(artifact.read_text(encoding="utf-8")) == vectors()

