from pathlib import Path

import pytest

from ukstress.provenance import ExperimentManifest, content_fingerprint, file_fingerprint


def test_content_fingerprint_is_order_independent_for_mappings() -> None:
    left = content_fingerprint({"а": 1, "б": [2, 3]}, namespace="test")
    right = content_fingerprint({"б": [2, 3], "а": 1}, namespace="test")

    assert left == right
    assert left.startswith("sha256:")


def test_fingerprint_namespace_separates_domains() -> None:
    assert content_fingerprint("same", namespace="a") != content_fingerprint(
        "same", namespace="b"
    )


def test_file_fingerprint_tracks_bytes(tmp_path: Path) -> None:
    path = tmp_path / "input.txt"
    path.write_text("замок", encoding="utf-8")
    first = file_fingerprint(path)
    path.write_text("за́мок", encoding="utf-8")

    assert file_fingerprint(path) != first


def _manifest() -> ExperimentManifest:
    return ExperimentManifest(
        run_id="run-1",
        command=["ukstress", "mine"],
        config_snapshot={"seed": 17},
        config_fingerprint="sha256:config",
        random_seeds={"python": 17},
        inputs={"corpus": "sha256:corpus", "lexicon": "sha256:lexicon"},
    )


def test_manifest_round_trip_and_refuses_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "manifest.yaml"
    manifest = _manifest()
    manifest.write(path)

    assert ExperimentManifest.read(path) == manifest
    with pytest.raises(FileExistsError):
        manifest.write(path)


def test_manifest_detects_tampering(tmp_path: Path) -> None:
    path = tmp_path / "manifest.yaml"
    _manifest().write(path)
    path.write_text(path.read_text(encoding="utf-8").replace("run-1", "run-2"), encoding="utf-8")

    with pytest.raises(ValueError, match="fingerprint mismatch"):
        ExperimentManifest.read(path)
