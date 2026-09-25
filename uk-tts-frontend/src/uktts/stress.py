"""Stage 2 — stress: mark the stressed vowel of every Ukrainian word.

Placement is served by the Go stress API, which owns the full four-tier
pipeline (lexicon, morphology, contextual cross-encoder, suffix and compound
fallbacks) and the PostgreSQL lexicon behind it. This module is a client; it
does not decide stress and must not, or the two would drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from .config import StressConfig


class StressUnavailable(RuntimeError):
    """The stress service could not be reached or is not ready to serve."""


@dataclass(frozen=True)
class StressToken:
    start: int
    end: int
    text: str
    output_text: str
    status: str
    candidates: tuple[str, ...] = ()
    margin: float | None = None

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> StressToken:
        return cls(
            start=int(payload["start"]),
            end=int(payload["end"]),
            text=str(payload["text"]),
            output_text=str(payload["output_text"]),
            status=str(payload["status"]),
            candidates=tuple(payload.get("candidates") or ()),
            margin=payload.get("margin"),
        )


@dataclass(frozen=True)
class StressResult:
    text: str
    tokens: tuple[StressToken, ...] = ()
    dataset_id: int | None = None
    model_version: str | None = None
    warnings: tuple[str, ...] = field(default=())


class Stresser(Protocol):
    name: str

    def stress(self, text: str, on_ambiguity: str | None = None,
               combiner: bool | None = None) -> StressResult: ...

    def ready(self) -> tuple[bool, str]: ...


class PassthroughStresser:
    """Return text unchanged — used when the stress stage is switched off."""

    name = "passthrough"

    def stress(self, text: str, on_ambiguity: str | None = None,
               combiner: bool | None = None) -> StressResult:
        return StressResult(text=text)

    def ready(self) -> tuple[bool, str]:
        return True, "stress stage disabled"


class HTTPStresser:
    """Call `POST /v1/stress` on the Go stress API."""

    name = "http"

    def __init__(self, config: StressConfig, client: Any | None = None) -> None:
        self.config = config
        if client is not None:
            self._client = client
        else:
            import httpx

            self._client = httpx.Client(base_url=config.base_url, timeout=config.timeout)

    def close(self) -> None:
        close = getattr(self._client, "close", None)
        if close is not None:
            close()

    def ready(self) -> tuple[bool, str]:
        try:
            response = self._client.get("/health/ready")
        except Exception as error:  # noqa: BLE001 - reported, not swallowed
            return False, f"{self.config.base_url} unreachable: {error}"
        if response.status_code != 200:
            return False, f"{self.config.base_url} not ready: HTTP {response.status_code}"
        return True, f"{self.config.base_url} ready"

    def stress(self, text: str, on_ambiguity: str | None = None,
               combiner: bool | None = None) -> StressResult:
        if not text.strip():
            return StressResult(text=text)
        body: dict[str, Any] = {"text": text, "on_ambiguity": on_ambiguity or self.config.on_ambiguity}
        if combiner is not None:
            # the stress API's learned combiner, per request; unset, its default
            body["combiner"] = combiner
        try:
            response = self._client.post("/v1/stress", json=body)
        except Exception as error:  # noqa: BLE001 - re-raised with the endpoint named
            raise StressUnavailable(f"stress API at {self.config.base_url} unreachable") from error
        if response.status_code != 200:
            raise StressUnavailable(
                f"stress API at {self.config.base_url} returned HTTP {response.status_code}: "
                f"{response.text[:200]}"
            )
        payload = response.json()
        return StressResult(
            text=str(payload["text"]),
            tokens=tuple(StressToken.from_payload(item) for item in payload.get("tokens", ())),
            dataset_id=payload.get("dataset_id"),
            model_version=payload.get("model_version"),
            warnings=tuple(payload.get("warnings") or ()),
        )


def load(config: StressConfig) -> Stresser:
    return HTTPStresser(config) if config.enabled else PassthroughStresser()
