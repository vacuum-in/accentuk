"""HTTP service wrapping the pipeline: `POST /v1/prepare`."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import Config
from .pipeline import Pipeline
from .stress import StressUnavailable


class PrepareRequest(BaseModel):
    text: str | None = Field(default=None, min_length=1)
    texts: list[str] | None = None
    on_ambiguity: str | None = Field(default=None, pattern="^(default|preserve)$")
    # the stress API's learned combiner; unset, the API's own default
    combiner: bool | None = None
    mode: str = Field(default="both", pattern="^(both|verbalize|stress)$")
    include_tokens: bool = False


class PrepareResponse(BaseModel):
    results: list[dict[str, Any]]


def create_app(config: Config | None = None, pipeline_instance: Pipeline | None = None) -> FastAPI:
    state: dict[str, Pipeline] = {}
    if pipeline_instance is not None:
        state["pipeline"] = pipeline_instance

    def pipeline() -> Pipeline:
        if "pipeline" not in state:
            state["pipeline"] = Pipeline(config or Config.from_env())
        return state["pipeline"]

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        # Load the checkpoint at startup so the first request is not the one
        # that pays for it, and so a bad path fails the deployment, not a user.
        pipeline()
        yield

    app = FastAPI(title="Ukrainian TTS text frontend", version="0.1.0", lifespan=lifespan)

    @app.get("/health/live")
    def live() -> dict[str, str]:
        return {"status": "live"}

    @app.get("/health/ready")
    def ready() -> dict[str, Any]:
        status = pipeline().ready()
        if not status["ready"]:
            raise HTTPException(status_code=503, detail=status)
        return status

    @app.post("/v1/prepare", response_model=PrepareResponse)
    def prepare(request: PrepareRequest) -> PrepareResponse:
        if request.texts is not None:
            inputs = list(request.texts)
        elif request.text is not None:
            inputs = [request.text]
        else:
            raise HTTPException(status_code=400, detail="provide `text` or `texts`")
        if not inputs:
            raise HTTPException(status_code=400, detail="provide `text` or `texts`")
        try:
            results = pipeline().prepare_many(inputs, request.on_ambiguity, request.mode, request.combiner)
        except StressUnavailable as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        payload = []
        for result in results:
            item = result.to_dict()
            if not request.include_tokens:
                item.pop("tokens", None)
            payload.append(item)
        return PrepareResponse(results=payload)

    return app
