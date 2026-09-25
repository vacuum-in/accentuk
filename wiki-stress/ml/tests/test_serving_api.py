import contextlib

import pytest
from fastapi.testclient import TestClient
from pytest import MonkeyPatch

from ukstress_ml import serving


def test_resolve_payload_is_bound_from_json_body(monkeypatch: MonkeyPatch) -> None:
    class FakeRuntime:
        model_version = "test-model"
        inventory_hash = "test-inventory"

        def resolve(self, targets: list[serving.Target]) -> list[dict[str, object]]:
            assert targets == [
                serving.Target(
                    sentence="ангара",
                    start=0,
                    end=6,
                    form="ангара",
                    candidates=("1", "2"),
                )
            ]
            return [
                {
                    "index": 0,
                    "signature": "2",
                    "scores": {"1": -2.0, "2": 1.0},
                    "margin": 3.0,
                    "status": "selected",
                }
            ]

    monkeypatch.setattr(
        serving, "ContextualStressModel", lambda *args, **kwargs: FakeRuntime()
    )
    client = TestClient(serving.create_app())

    response = client.post(
        "/internal/v1/resolve",
        json={
            "model_version": "test-model",
            "inventory_hash": "test-inventory",
            "targets": [
                {
                    "sentence": "ангара",
                    "start": 0,
                    "end": 6,
                    "form": "ангара",
                    "candidates": ["1", "2"],
                }
            ],
        },
    )

    assert response.status_code == 200
    assert response.json()["decisions"][0]["signature"] == "2"


# The readiness canary. Its whole purpose is the failure that raises nothing:
# uk_core_news_trf 3.7.2 without its curated-transformers factory loads, tags
# every token featurelessly, resolves zero, and costs 17 points in silence.

def test_the_canary_reports_ok_when_the_tier_resolves(monkeypatch) -> None:
    from ukstress_ml import serving

    class Parser:
        def parse(self, sentence):
            return {(s, e): ("NOUN", "Case=Nom|Number=Plur")
                    for _, _, s, e in serving.MORPHOLOGY_CANARY}

    monkeypatch.setattr(type(serving._morphology), "available",
                        property(lambda self: True))
    monkeypatch.setattr(serving._morphology, "acquire",
                        lambda: contextlib.nullcontext(Parser()))
    monkeypatch.setattr(serving._morphology, "readings",
                        lambda form, parser: [(["Case=Nom", "Number=Plur"], [2]),
                                              (["Case=Gen", "Number=Sing"], [4])])
    probe = serving.morphology_canary()
    assert probe["ok"] is True
    assert probe["resolved"] == probe["probed"]


def test_a_tagger_that_returns_no_features_is_caught(monkeypatch) -> None:
    """The exact 3.7.2 signature: a parse arrives, carrying nothing."""
    from ukstress_ml import serving

    class Featureless:
        def parse(self, sentence):
            return {(s, e): ("", "") for _, _, s, e in serving.MORPHOLOGY_CANARY}

    monkeypatch.setattr(type(serving._morphology), "available",
                        property(lambda self: True))
    monkeypatch.setattr(serving._morphology, "acquire",
                        lambda: contextlib.nullcontext(Featureless()))
    monkeypatch.setattr(serving._morphology, "readings",
                        lambda form, parser: [(["Case=Nom"], [2]), (["Case=Gen"], [4])])
    probe = serving.morphology_canary()
    assert probe["ok"] is False
    assert probe["resolved"] == 0
    assert "resolves nothing" in probe["detail"]


def test_an_unavailable_tier_is_caught(monkeypatch) -> None:
    from ukstress_ml import serving

    monkeypatch.setattr(type(serving._morphology), "available",
                        property(lambda self: False))
    assert serving.morphology_canary()["ok"] is False


def test_a_probe_that_raises_does_not_escape(monkeypatch) -> None:
    from ukstress_ml import serving

    class Broken:
        def parse(self, sentence):
            raise RuntimeError("pool exploded")

    monkeypatch.setattr(type(serving._morphology), "available",
                        property(lambda self: True))
    monkeypatch.setattr(serving._morphology, "acquire",
                        lambda: contextlib.nullcontext(Broken()))
    probe = serving.morphology_canary()
    assert probe["ok"] is False
    assert "probe failed" in probe["detail"]


# Tier 3 as classification, serving the same contract as the pair model.

def test_load_model_picks_the_backend_from_the_checkpoint(tmp_path, monkeypatch) -> None:
    """A classifier checkpoint carries head.pt; a pair checkpoint does not."""
    import json as _json

    from ukstress_ml import serving

    manifest = tmp_path / "serving_manifest.json"
    manifest.write_text(_json.dumps({
        "model_version": "test", "inventory_hash": "h", "threshold": 0.5, "forms": {},
    }), encoding="utf-8")
    (tmp_path / "checkpoint").mkdir()

    seen = {}
    monkeypatch.setattr(serving, "ContextualStressModel",
                        lambda d, m: seen.setdefault("kind", "pair"))
    monkeypatch.setattr(serving, "ClassifierStressModel",
                        lambda d, m: seen.setdefault("kind", "classifier"))

    serving.load_model(str(tmp_path), str(manifest))
    assert seen["kind"] == "pair"

    seen.clear()
    (tmp_path / "checkpoint" / "head.pt").write_bytes(b"")
    serving.load_model(str(tmp_path), str(manifest))
    assert seen["kind"] == "classifier"


def test_validate_target_rejects_a_candidate_set_the_manifest_disagrees_with() -> None:
    """An inventory-version mismatch must fail, not resolve against stale data."""
    from ukstress_ml.serving import ServingError, Target, validate_target

    manifest = {"forms": {"замок": {"signatures": ["0", "1"], "candidates": []}}}
    good = Target(sentence="Тут замок.", start=4, end=9, form="замок",
                  candidates=("0", "1"))
    _entry, allowed = validate_target(good, manifest)
    assert allowed == ["0", "1"]

    stale = Target(sentence="Тут замок.", start=4, end=9, form="замок",
                   candidates=("0", "1", "2"))
    with pytest.raises(ServingError, match="differ from model inventory"):
        validate_target(stale, manifest)


def test_validate_target_rejects_offsets_that_miss_the_form() -> None:
    from ukstress_ml.serving import ServingError, Target, validate_target

    manifest = {"forms": {"замок": {"signatures": ["0", "1"], "candidates": []}}}
    wrong = Target(sentence="Тут замок.", start=0, end=3, form="замок",
                   candidates=("0", "1"))
    with pytest.raises(ServingError, match="do not identify"):
        validate_target(wrong, manifest)


def test_the_span_helper_falls_back_when_truncation_removed_the_target() -> None:
    from ukstress_ml.serving import ClassifierStressModel

    span = ClassifierStressModel._span(
        None, [(0, 0), (0, 3), (3, 7), (0, 0)], [1, 1, 1, 0], 900, 905)
    assert span == [True, True, True, False]


def test_the_span_helper_covers_only_the_target() -> None:
    from ukstress_ml.serving import ClassifierStressModel

    span = ClassifierStressModel._span(
        None, [(0, 0), (0, 3), (4, 9), (9, 10), (0, 0)], [1, 1, 1, 1, 0], 4, 9)
    assert span == [False, False, True, False, False]
