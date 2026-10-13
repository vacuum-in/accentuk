import json
from pathlib import Path
import pytest

from accentuk.assets import AssetBundle, get_releases_descriptor
from accentuk.errors import AssetError


def test_releases_descriptor():
    desc = get_releases_descriptor()
    assert "data_release" in desc
    assert "models" in desc
    assert desc["dataset_id"] == 2
    assert desc["normalization_version"] == "ukstress-nfd-v1"


def test_asset_bundle_open_validation(tmp_path: Path):
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()

    # Missing manifest.json
    with pytest.raises(AssetError, match="manifest missing"):
        AssetBundle.open(bundle_dir)

    # Valid manifest and files
    (bundle_dir / "lexicon.marisa").write_bytes(b"content")
    import hashlib
    sha = hashlib.sha256(b"content").hexdigest()

    manifest = {
        "schema_version": 1,
        "normalization_version": "ukstress-nfd-v1",
        "dataset_id": 2,
        "inventory_hash": "testhash",
        "files": {"lexicon.marisa": sha},
    }
    (bundle_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    bundle = AssetBundle.open(bundle_dir)
    assert bundle.dataset_id == 2
    assert bundle.lexicon_path == bundle_dir / "lexicon.marisa"

    # Corrupt checksum
    (bundle_dir / "lexicon.marisa").write_bytes(b"tampered")
    with pytest.raises(AssetError, match="Corrupted bundle file"):
        AssetBundle.open(bundle_dir)


def test_asset_bundle_incompatible_version(tmp_path: Path):
    bundle_dir = tmp_path / "bundle_bad_ver"
    bundle_dir.mkdir()

    manifest = {
        "schema_version": 2,  # incompatible
        "normalization_version": "ukstress-nfd-v1",
        "dataset_id": 2,
        "files": {},
    }
    (bundle_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(AssetError, match="Incompatible bundle schema version"):
        AssetBundle.open(bundle_dir)


def test_download_assets_with_bad_checksum(tmp_path: Path, monkeypatch):
    import urllib.request
    from unittest.mock import MagicMock
    from accentuk.assets import download_assets, _download_file

    # Mock urllib.request.urlopen returning dummy data
    mock_resp = MagicMock()
    mock_resp.read.side_effect = [b"mock content", b""]
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.__exit__.return_value = None

    monkeypatch.setattr(urllib.request, "urlopen", lambda req: mock_resp)

    # When expected_sha does not match "mock content", AssetError must be raised
    dest = tmp_path / "test.file"
    with pytest.raises(AssetError, match="Checksum mismatch"):
        _download_file("http://example.com/file", dest, expected_sha="deadbeef")

    # Destination file should not have been created (cleaned up)
    assert not dest.exists()

