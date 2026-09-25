from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pytest

from uktts.config import StressConfig
from uktts.stress import HTTPStresser, StressUnavailable


@dataclass
class FakeResponse:
    status_code: int
    payload: Any = None
    text: str = ""

    def json(self) -> Any:
        return self.payload


class FakeClient:
    def __init__(self, response: Any = None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.requests: list[tuple[str, dict[str, Any]]] = []

    def post(self, url: str, json: dict[str, Any]) -> Any:  # noqa: A002 - httpx's name
        if self.error is not None:
            raise self.error
        self.requests.append((url, json))
        return self.response

    def get(self, url: str) -> Any:
        if self.error is not None:
            raise self.error
        return self.response


PAYLOAD = {
    "text": "нови́й дім",
    "tokens": [
        {
            "start": 0,
            "end": 5,
            "text": "новий",
            "output_text": "нови́й",
            "status": "stressed",
            "candidates": ["нови́й", "но́вий"],
            "margin": 0.42,
        },
        {"start": 6, "end": 9, "text": "дім", "output_text": "дім", "status": "not_required"},
    ],
    "dataset_id": 7,
    "model_version": "v19-v10",
    "warnings": [],
}


def stresser(client: FakeClient, **overrides: Any) -> HTTPStresser:
    return HTTPStresser(StressConfig(**overrides), client=client)


def test_a_successful_response_is_parsed_into_tokens() -> None:
    client = FakeClient(FakeResponse(200, PAYLOAD))
    result = stresser(client).stress("новий дім")
    assert result.text == "нови́й дім"
    assert result.dataset_id == 7
    assert result.model_version == "v19-v10"
    assert result.tokens[0].candidates == ("нови́й", "но́вий")
    assert result.tokens[1].status == "not_required"


def test_the_configured_ambiguity_policy_is_sent() -> None:
    client = FakeClient(FakeResponse(200, PAYLOAD))
    stresser(client, on_ambiguity="preserve").stress("новий дім")
    assert client.requests[0][1]["on_ambiguity"] == "preserve"


def test_a_per_call_policy_overrides_the_configured_one() -> None:
    client = FakeClient(FakeResponse(200, PAYLOAD))
    stresser(client, on_ambiguity="default").stress("новий дім", on_ambiguity="preserve")
    assert client.requests[0][1]["on_ambiguity"] == "preserve"


def test_blank_text_is_not_sent_to_the_service() -> None:
    client = FakeClient(FakeResponse(200, PAYLOAD))
    assert stresser(client).stress("   ").text == "   "
    assert client.requests == []


def test_an_error_status_names_the_endpoint() -> None:
    client = FakeClient(FakeResponse(503, None, json.dumps({"code": "unavailable"})))
    with pytest.raises(StressUnavailable, match="HTTP 503"):
        stresser(client, base_url="http://stress:8080").stress("новий дім")


def test_a_transport_failure_is_reported_as_unavailable() -> None:
    client = FakeClient(error=OSError("connection refused"))
    with pytest.raises(StressUnavailable, match="unreachable"):
        stresser(client).stress("новий дім")


def test_readiness_reflects_the_health_endpoint() -> None:
    assert stresser(FakeClient(FakeResponse(200, {}))).ready()[0] is True
    assert stresser(FakeClient(FakeResponse(503, {}))).ready()[0] is False
    assert stresser(FakeClient(error=OSError("down"))).ready()[0] is False


def test_combiner_is_sent_only_when_asked() -> None:
    client = FakeClient(FakeResponse(200, PAYLOAD))
    stresser(client).stress("новий дім")
    assert "combiner" not in client.requests[0][1]
    stresser(client).stress("новий дім", combiner=True)
    assert client.requests[1][1]["combiner"] is True
