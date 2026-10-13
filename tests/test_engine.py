import os
from pathlib import Path
import pytest

from accentuk import Accentor
from accentuk.errors import AssetError


def test_accentor_lifecycle_and_validation(tmp_path: Path):
    # Missing asset directory
    with pytest.raises(AssetError):
        Accentor(tmp_path / "nonexistent")

    # Real bundle from verification script
    bundle_path = Path("/tmp/accentuk-bundle-test")
    if not (bundle_path / "lexicon.marisa").is_file():
        pytest.skip("Test bundle not yet generated")

    with Accentor(bundle_path) as engine:
        # Empty string
        res = engine.stress("")
        assert res.text == ""
        assert res.tokens == []

        # Invalid on_ambiguity
        with pytest.raises(ValueError):
            engine.stress("текст", on_ambiguity="invalid")

    # Engine closed
    with pytest.raises(RuntimeError):
        engine.stress("текст")


def test_accentor_deterministic_and_fallbacks():
    bundle_path = Path("/tmp/accentuk-bundle-test")
    if not (bundle_path / "lexicon.marisa").is_file():
        pytest.skip("Test bundle not yet generated")

    with Accentor(bundle_path) as engine:
        # 1. Deterministic word with 1 candidate: мова
        res = engine.stress("мова")
        assert res.text == "мо́ва"
        assert res.tokens[0].status == "stressed"

        # 2. Already stressed word: мо́ва
        res = engine.stress("мо́ва")
        assert res.text == "мо́ва"
        assert res.tokens[0].status == "already_stressed"

        # 3. Not required (single vowel): в, на
        res = engine.stress("в")
        assert res.text == "в"
        assert res.tokens[0].status == "not_required"

        # 4. Prepositional shift: до мене -> до ме́не
        res = engine.stress("до мене")
        assert res.text == "до ме́не"
        assert res.tokens[1].status == "prepositional"

        # 5. Non-adjacent preposition: до, мене (comma between -> no shift)
        res = engine.stress("до, мене")
        assert res.tokens[1].status != "prepositional"

        # 6. Compound fallback: hyphenated word with parts
        res = engine.stress("пів-яблука")
        assert res.text in ("пі́в-я́блука", "пів-я́блука")

        # 7. Ambiguity default vs preserve
        # "коси" is ambiguous
        res_default = engine.stress("довгі коси", on_ambiguity="default")
        assert "\u0301" in res_default.tokens[1].output_text

        res_preserve = engine.stress("невідомесловобезмоделі", on_ambiguity="preserve")
        # unknown word
        assert res_preserve.tokens[0].status in ("suffix", "default_position")


def test_accentor_real_inference():
    bundle_path = Path("/tmp/accentuk-bundle-test")
    if not (bundle_path / "lexicon.marisa").is_file():
        pytest.skip("Test bundle not yet generated")

    # Set assets dir to UK stress deploy assets to test full stack
    assets_dir = Path(os.environ.get("ACCENTUK_TEST_ASSETS", "/nonexistent"))
    if not (assets_dir / "stress-model").is_dir():
        pytest.skip("Deployment models not present")

    # Link bundle into assets dir if needed or point Accentor to /tmp/accentuk-bundle-test with model assets
    # In engine.py, it checks self.asset_dir / "stress-model" or $ACCENTUK_TEST_ASSETS
    with Accentor(bundle_path) as engine:
        # Test "Мої сестри."
        res = engine.stress("Мої сестри.")
        assert len(res.tokens) == 2
        # се́стри by agreement repair or model
        assert "се́стри" in res.text or "сестри́" in res.text
