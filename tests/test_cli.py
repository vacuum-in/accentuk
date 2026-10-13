import pytest
from pathlib import Path
from accentuk.cli import main


def test_cli_help(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert "usage:" in captured.out or "accentuk" in captured.out


def test_cli_subcommand_help(capsys):
    for subcmd in ["download", "export", "stress", "prepare"]:
        with pytest.raises(SystemExit) as exc:
            main([subcmd, "--help"])
        assert exc.value.code == 0


def test_cli_stress_missing_assets(capsys):
    ret = main(["stress", "Привіт", "--assets", "/nonexistent/bundle/dir"])
    assert ret == 1
    captured = capsys.readouterr()
    assert "Error:" in captured.err


def test_cli_stress_success(capsys):
    bundle_path = Path("/tmp/accentuk-bundle-test")
    if not (bundle_path / "lexicon.marisa").is_file():
        pytest.skip("Test bundle fixture not present")

    ret = main(["stress", "Мова.", "--assets", str(bundle_path)])
    assert ret == 0
    captured = capsys.readouterr()
    assert "Мо́ва." in captured.out


def test_cli_stress_preserve_ambiguity(capsys):
    bundle_path = Path("/tmp/accentuk-bundle-test")
    if not (bundle_path / "lexicon.marisa").is_file():
        pytest.skip("Test bundle fixture not present")

    # "сестри" is ambiguous without ML context (се́стри vs сестри́)
    ret = main(["stress", "сестри", "--ambiguity", "preserve", "--assets", str(bundle_path)])
    assert ret == 0
    captured = capsys.readouterr()
    assert "сестри" in captured.out.strip()


def test_cli_prepare_success(capsys):
    bundle_path = Path("/tmp/accentuk-bundle-test")
    if not (bundle_path / "lexicon.marisa").is_file():
        pytest.skip("Test bundle fixture not present")

    # Simple prepare without verbalizer requirement
    ret = main(["prepare", "Мова.", "--assets", str(bundle_path)])
    assert ret == 0
    captured = capsys.readouterr()
    assert "Мо́ва." in captured.out
