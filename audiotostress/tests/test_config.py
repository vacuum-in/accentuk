from pathlib import Path

import pytest
from pydantic import ValidationError

from ukstress.config import load_config


def test_load_config_expands_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("UKSTRESS_TEST_MODEL", "example/encoder")
    path = tmp_path / "config.yaml"
    path.write_text(
        "features:\n  ssl_model: ${UKSTRESS_TEST_MODEL}\nmodel:\n  hidden_size: 128\n",
        encoding="utf-8",
    )

    config = load_config(path)

    assert config.features.ssl_model == "example/encoder"
    assert config.sample_rate == 16_000


def test_load_config_rejects_unknown_keys(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("surprise: true\n", encoding="utf-8")

    with pytest.raises(ValidationError, match="surprise"):
        load_config(path)


def test_load_config_rejects_incompatible_attention_shape(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("model:\n  hidden_size: 255\n  attention_heads: 4\n", encoding="utf-8")

    with pytest.raises(ValidationError, match="divisible"):
        load_config(path)


def test_load_config_rejects_unset_environment(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("features:\n  ssl_model: ${UKSTRESS_DOES_NOT_EXIST}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="UKSTRESS_DOES_NOT_EXIST"):
        load_config(path)
