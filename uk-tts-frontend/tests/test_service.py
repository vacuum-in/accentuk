from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from tests.test_pipeline import FakeVerbalizer, RecordingStresser, build  # noqa: E402
from uktts.service import create_app  # noqa: E402
from uktts.stress import StressResult, StressUnavailable  # noqa: E402


def client(stresser=None) -> TestClient:
    pipeline = build(FakeVerbalizer(), stresser or RecordingStresser())
    return TestClient(create_app(pipeline_instance=pipeline))


def test_prepare_returns_the_prepared_text() -> None:
    response = client().post("/v1/prepare", json={"text": "Абрикос."})
    assert response.status_code == 200
    result = response.json()["results"][0]
    assert result["text"] == "А́БРИКОС."
    assert result["verbalized"] == "АБРИКОС."
    assert "tokens" not in result


def test_tokens_are_returned_only_when_asked_for() -> None:
    response = client().post("/v1/prepare", json={"text": "Абрикос.", "include_tokens": True})
    assert response.json()["results"][0]["tokens"][0]["status"] == "stressed"


def test_a_batch_returns_one_result_per_input_in_order() -> None:
    response = client().post("/v1/prepare", json={"texts": ["Перше.", "Друге."]})
    assert [item["verbalized"] for item in response.json()["results"]] == ["ПЕРШЕ.", "ДРУГЕ."]


def test_a_request_without_text_is_rejected() -> None:
    assert client().post("/v1/prepare", json={}).status_code == 400


def test_an_invalid_ambiguity_policy_is_rejected() -> None:
    response = client().post("/v1/prepare", json={"text": "Слово.", "on_ambiguity": "guess"})
    assert response.status_code == 422


class DownStresser(RecordingStresser):
    def stress(self, text: str, on_ambiguity: str | None = None) -> StressResult:
        raise StressUnavailable("stress API unreachable")

    def ready(self) -> tuple[bool, str]:
        return False, "stress API unreachable"


def test_a_stress_outage_is_reported_as_503_not_as_unstressed_text() -> None:
    response = client(DownStresser()).post("/v1/prepare", json={"text": "Слово."})
    assert response.status_code == 503


def test_readiness_fails_while_the_stress_service_is_down() -> None:
    down = client(DownStresser())
    assert down.get("/health/live").json() == {"status": "live"}
    assert down.get("/health/ready").status_code == 503
    assert client().get("/health/ready").json()["ready"] is True


def test_mode_is_passed_through_to_the_pipeline() -> None:
    response = client().post("/v1/prepare", json={"text": "Абрикос.", "mode": "verbalize"})
    assert response.status_code == 200
    assert response.json()["results"][0]["text"] == "АБРИКОС."


def test_an_unknown_mode_is_rejected_by_the_api() -> None:
    response = client().post("/v1/prepare", json={"text": "Слово.", "mode": "phonemize"})
    assert response.status_code == 422
