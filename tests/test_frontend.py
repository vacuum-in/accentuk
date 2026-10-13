import os
from pathlib import Path
import pytest

from accentuk import Accentor, Frontend


def test_frontend_lifecycle_and_ownership():
    bundle_path = Path("/tmp/accentuk-bundle-test")
    if not (bundle_path / "lexicon.marisa").is_file():
        pytest.skip("Test bundle not yet generated")

    # 1. Self-owned engine is closed on Frontend.close()
    with Frontend(bundle_path, verbalize=False) as fe:
        res = fe.prepare("Мова.")
        assert res.text == "Мо́ва."
        assert res.verbalized == "Мова."
        assert res.source == "Мова."

    # 2. Caller-owned engine is NOT closed on Frontend.close()
    with Accentor(bundle_path) as engine:
        fe = Frontend(engine=engine, verbalize=False)
        res = fe.prepare("Мова.")
        assert res.text == "Мо́ва."
        fe.close()
        # Engine should still be usable!
        assert engine.stress("Мова.").text == "Мо́ва."


def test_frontend_real_verbalizer():
    bundle_path = Path("/tmp/accentuk-bundle-test")
    verbalizer_path = (Path(os.environ.get("ACCENTUK_TEST_ASSETS", "/nonexistent")) / "verbalizer")
    if not (bundle_path / "lexicon.marisa").is_file() or not (verbalizer_path / "config.json").is_file():
        pytest.skip("Assets not present")

    with Frontend(bundle_path, verbalize=True) as fe:
        # Plain text bypasses verbalizer
        res = fe.prepare("Сидів спокійно.")
        assert "спокійно" in res.verbalized

        # Text with date/number verbalizes
        res = fe.prepare("2024 р.")
        assert res.verbalized != "2024 р."
        assert len(res.tokens) > 0
        # Verify offsets slice verbalized text
        for tok in res.tokens:
            assert res.verbalized[tok.start : tok.end] == tok.text
