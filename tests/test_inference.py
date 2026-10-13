import os
import pytest
from pathlib import Path

from accentuk.errors import ServingError
from accentuk.inference import ContextualStressModel, Target, validate_target

MANIFEST_FIXTURE = {
    "model_version": "v3-test",
    "inventory_hash": "testhash",
    "threshold": 1.5,
    "forms": {
        "замок": {
            "signatures": ["0", "1"],
            "candidates": [
                {"signature": "0", "stressed": "за́мок", "definition": "fortress"},
                {"signature": "1", "stressed": "замо́к", "definition": "device"},
            ],
        }
    },
}


def test_validate_target_errors():
    manifest = MANIFEST_FIXTURE

    # Offsets outside sentence
    with pytest.raises(ServingError, match="outside the sentence"):
        validate_target(Target("Це замок.", -1, 5, "замок", ("0", "1")), manifest)
    with pytest.raises(ServingError, match="outside the sentence"):
        validate_target(Target("Це замок.", 3, 20, "замок", ("0", "1")), manifest)

    # Form mismatch
    with pytest.raises(ServingError, match="do not identify"):
        validate_target(Target("Це замок.", 0, 2, "замок", ("0", "1")), manifest)

    # Form outside coverage
    with pytest.raises(ServingError, match="outside frozen training coverage"):
        validate_target(Target("Це слово.", 3, 8, "слово", ("0",)), manifest)

    # Candidates mismatch
    with pytest.raises(ServingError, match="differ from model inventory"):
        validate_target(Target("Це замок.", 3, 8, "замок", ("0",)), manifest)

    # Valid target
    entry, allowed = validate_target(
        Target("Це замок.", 3, 8, "замок", ("0", "1")), manifest
    )
    assert allowed == ["0", "1"]
    assert entry["signatures"] == ["0", "1"]


def test_real_crossencoder_inference():
    model_dir = (Path(os.environ.get("ACCENTUK_TEST_ASSETS", "/nonexistent")) / "stress-model")
    if not (model_dir / "checkpoint").is_dir():
        pytest.skip("Local model fixture not present")

    model = ContextualStressModel(model_dir)
    version, inv_hash = model.metadata()
    assert version != ""
    assert inv_hash != ""

    target = Target(
        sentence="Це замок на горі.",
        start=3,
        end=8,
        form="замок",
        candidates=("0", "1"),
    )
    decisions = model.resolve([target])
    assert len(decisions) == 1
    d = decisions[0]
    assert d["index"] == 0
    assert d["signature"] in ("0", "1")
    assert "scores" in d
    assert "0" in d["scores"] and "1" in d["scores"]
    assert d["status"] in ("selected", "low_margin")
