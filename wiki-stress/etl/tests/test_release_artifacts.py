import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_real_staging_manifest_has_published_inventory_counts() -> None:
    manifest = json.loads(
        (ROOT / "output/9dacc408065a68ee-forms-v4/manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["statistics"]["lexemes"] == 29_579
    assert manifest["statistics"]["word_forms"] == 363_272
    assert manifest["statistics"]["stress_variants"] == 364_130


def test_contextual_serving_manifest_is_frozen_and_release_eligible() -> None:
    path = ROOT / "models/v3-xenc/serving_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    report = json.loads(
        (ROOT / "models/v3-xenc/release_report.json").read_text(encoding="utf-8")
    )
    assert report["release_eligible"] is True
    assert manifest["model_version"] == "v3-xenc"
    assert manifest["inventory_hash"] == report["inventory_hash"]
    assert manifest["threshold"] == report["abstention"]["threshold"]
    assert len(manifest["forms"]) == 570
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "a7bd1285f51530d46a2aa3870eadeb6c7b1adb39bdb8052e5d7106cf0ed83faf"
    )
