"""Test Full and Light mode execution in accentuk."""

import os
from pathlib import Path
import pytest

from accentuk import Accentor, Frontend
from accentuk.inference import Target, CT2ContextualStressModel
from accentuk.token_resolver import CT2TokenResolver
from accentuk.verbalize import CT2GemmaVerbalizer, CT2MarianVerbalizer, load_verbalizer


@pytest.fixture(scope="module")
def bundle_path():
    p = Path("/tmp/accentuk-bundle-test")
    if not (p / "lexicon.marisa").is_file():
        pytest.skip("Test bundle fixture not present")
    return p


def test_accentor_mode_validation(bundle_path: Path):
    with pytest.raises(ValueError, match="mode must be 'auto', 'light', or 'full'"):
        Accentor(asset_dir=bundle_path, mode="invalid_mode")

    with Accentor(asset_dir=bundle_path, mode="auto") as acc:
        assert acc.mode == "auto"


def test_accentor_light_mode(bundle_path: Path):
    with Accentor(asset_dir=bundle_path, mode="light") as acc:
        assert acc.mode == "light"
        # In light mode, heavy spaCy TRF and combiner tiers must be bypassed
        assert acc._ensure_spacy_tier() is None
        assert acc._ensure_combiner() == (None, None)

        # Basic dictionary word
        res = acc.stress("Мова.")
        assert res.text == "Мо́ва."
        assert res.mode == "light"

        # Heteronym disambiguation in light mode
        res_het = acc.stress("Замок стоїть на високій горі.")
        assert res_het.mode == "light"
        assert len(res_het.tokens) > 0


def test_accentor_full_mode(bundle_path: Path):
    with Accentor(asset_dir=bundle_path, mode="full") as acc:
        assert acc.mode == "full"
        res = acc.stress("Мова.")
        assert res.text == "Мо́ва."
        assert res.mode == "full"


def test_ct2_verbalizer():
    ct2_dir = (Path(os.environ.get("ACCENTUK_TEST_CT2", "/nonexistent")) / "verbalizer-ct2-int8")
    if not (ct2_dir / "model.bin").is_file():
        pytest.skip("verbalizer-ct2-int8 model not found")

    verb = CT2MarianVerbalizer(ct2_dir)
    res = verb.verbalize(["У 1991 році", "25 яблук"])
    assert len(res) == 2
    assert "тисяча дев'ятсот" in res[0]
    assert "двадцять п'ять" in res[1]


def test_ct2_crossencoder():
    ct2_dir = (Path(os.environ.get("ACCENTUK_TEST_CT2", "/nonexistent")) / "stress-model-ct2-int8")
    manifest = (Path(os.environ.get("ACCENTUK_TEST_ASSETS", "/nonexistent")) / "stress-model/serving_manifest.json")
    if not (ct2_dir / "model.bin").is_file() or not manifest.is_file():
        pytest.skip("stress-model-ct2-int8 or manifest not found")

    model = CT2ContextualStressModel(ct2_dir, manifest_path=manifest)
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
    assert d["status"] in ("selected", "low_margin")


def test_ct2_token_resolver():
    ct2_dir = (Path(os.environ.get("ACCENTUK_TEST_CT2", "/nonexistent")) / "tok-v5-ct2-int8")
    if not (ct2_dir / "model.bin").is_file():
        pytest.skip("tok-v5-ct2-int8 model not found")

    resolver = CT2TokenResolver(ct2_dir)
    sample_target = {
        "index": 0,
        "sentence": "Мої сестри приїхали додому.",
        "start": 4,
        "end": 10,
        "form": "сестри",
        "candidates": ["0", "1"],
    }
    decisions = resolver.resolve([sample_target], with_probabilities=True, ignore_coverage=True)
    assert len(decisions) == 1
    assert decisions[0]["signature"] == "0"
    assert decisions[0]["confidence"] > 0.9


def test_frontend_modes(bundle_path: Path):
    with Frontend(asset_dir=bundle_path, mode="light") as frontend:
        assert frontend.mode == "light"
        res = frontend.prepare("Вона купила 25 яблук.")
        assert res.mode == "light"
        assert len(res.text) > 0


def test_gemma_verbalizer():
    gemma_dir = Path(os.environ.get("ACCENTUK_TEST_GEMMA", "/nonexistent"))
    if not (gemma_dir / "model.bin").is_file():
        pytest.skip("gemma-verbalizer-ct2-int8 not found")
    verb = load_verbalizer(gemma_dir, mode="light")
    assert isinstance(verb, CT2GemmaVerbalizer)
    out = verb.verbalize(["Зустріч о 08:05 у кабінеті 12.", "0000000", "Збитки оцінили в 7000000000 доларів.", ""])
    assert out[0] == "Зустріч о восьмій нуль п'ять у кабінеті дванадцять."
    assert out[1] == "нуль нуль нуль нуль нуль нуль нуль"
    assert "сім мільярдів" in out[2]
    assert out[3] == ""
    assert not any(ch.isdigit() for o in out for ch in o)
