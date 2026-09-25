import json
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import psycopg
import pytest

from ukstress.cli import main
from ukstress.downloader import DumpManifest


def test_download_command_emits_manifest(tmp_path: Path, capsys: object) -> None:
    output = tmp_path / "dump.bz2"
    manifest = DumpManifest("https://example.invalid/dump", output.name, "0" * 64, "now", 10)
    with patch("ukstress.cli.download_dump", return_value=manifest) as download:
        assert (
            main(
                [
                    "download",
                    "--url",
                    manifest.url,
                    "--output",
                    str(output),
                    "--sha256",
                    manifest.sha256,
                ]
            )
            == 0
        )
    download.assert_called_once_with(manifest.url, output, manifest.sha256)
    captured = capsys.readouterr()  # type: ignore[attr-defined]
    assert json.loads(captured.out)["sha256"] == "0" * 64


def test_command_failure_has_nonzero_exit_code(capsys: object) -> None:
    with patch("ukstress.cli.download_dump", side_effect=ValueError("bad checksum")):
        assert main(["download"]) == 1
    captured = capsys.readouterr()  # type: ignore[attr-defined]
    assert "bad checksum" in captured.err


def test_database_error_is_reported_without_a_raw_traceback(capsys: object) -> None:
    # psycopg.Error is not an OSError/RuntimeError/ValueError, so without an
    # explicit handler it would fall through to the default uncaught-exception
    # path and print a full Python traceback instead of a clean CLI message.
    with patch(
        "ukstress.cli.publish_dataset",
        side_effect=psycopg.OperationalError("connection to server failed"),
    ):
        assert main(["publish", "--database-url", "postgresql://x/y", "1"]) == 1
    captured = capsys.readouterr()  # type: ignore[attr-defined]
    assert "ukstress: database error: connection to server failed" in captured.err
    assert "Traceback" not in captured.err


def test_migrate_command_reports_applied_migrations(
    monkeypatch: pytest.MonkeyPatch, capsys: object, tmp_path: Path
) -> None:
    monkeypatch.setattr("ukstress.cli.apply_migrations", lambda *_args, **_kwargs: ["001"])
    assert (
        main(
            [
                "migrate",
                "--database-url",
                "postgresql://unused",
                "--migrations",
                str(tmp_path),
            ]
        )
        == 0
    )
    assert '"applied": ["001"]' in capsys.readouterr().out  # type: ignore[attr-defined]


@pytest.mark.parametrize("command", ["publish", "rollback"])
def test_dataset_activation_commands(
    command: str, monkeypatch: pytest.MonkeyPatch, capsys: object
) -> None:
    monkeypatch.setattr(f"ukstress.cli.{command}_dataset", lambda *_args: 4)
    assert main([command, "--database-url", "postgresql://unused", "5"]) == 0
    assert '"previous": 4' in capsys.readouterr().out  # type: ignore[attr-defined]


def test_cleanup_command_reports_deleted_datasets(
    monkeypatch: pytest.MonkeyPatch, capsys: object
) -> None:
    monkeypatch.setattr(
        "ukstress.cli.cleanup_retained_datasets", lambda *_args, **_kwargs: [1, 2]
    )
    assert main(["cleanup", "--database-url", "postgresql://unused", "--retain-count", "3"]) == 0
    output = capsys.readouterr().out  # type: ignore[attr-defined]
    assert '"deleted": [1, 2]' in output
    assert '"retain_count": 3' in output


def test_download_retries_resumable_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output = tmp_path / "dump.bz2"
    manifest = DumpManifest("https://example.invalid", output.name, "0" * 64, "now", 1)
    calls: Iterator[OSError | DumpManifest] = iter([OSError("temporary"), manifest])

    def attempt(*_args: object) -> DumpManifest:
        result = next(calls)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr("ukstress.cli.download_dump", attempt)
    assert main(["download", "--output", str(output), "--retries", "1"]) == 0


def test_keyboard_interrupt_has_shell_exit_code(
    monkeypatch: pytest.MonkeyPatch, capsys: object
) -> None:
    def interrupt(*_args: object) -> DumpManifest:
        raise KeyboardInterrupt

    monkeypatch.setattr("ukstress.cli.download_dump", interrupt)
    assert main(["download"]) == 130
    assert "interrupted" in capsys.readouterr().err  # type: ignore[attr-defined]


def test_help_lists_lifecycle_commands(capsys: object) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])
    assert exit_info.value.code == 0
    output = capsys.readouterr().out  # type: ignore[attr-defined]
    assert "report-unhandled" in output
    assert "export" in output
