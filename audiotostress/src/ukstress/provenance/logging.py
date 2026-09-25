"""Structured JSON logging with pipeline context."""

from __future__ import annotations

import contextlib
import contextvars
import json
import logging
import sys
from collections.abc import Iterator, Mapping
from datetime import UTC, datetime
from typing import TextIO

_CONTEXT_NAMES = ("run_id", "shard_id", "source_id", "record_id")
_LOG_CONTEXT: contextvars.ContextVar[dict[str, str] | None] = contextvars.ContextVar(
    "ukstress_log_context", default=None
)


class JsonFormatter(logging.Formatter):
    """Serialize one log record per line for machine ingestion."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        payload.update(_LOG_CONTEXT.get() or {})
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def configure_logging(*, level: int = logging.INFO, stream: TextIO | None = None) -> None:
    """Configure the package root logger with a single JSON handler."""

    logger = logging.getLogger("ukstress")
    logger.handlers.clear()
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False


@contextlib.contextmanager
def bind_log_context(**values: str | None) -> Iterator[None]:
    """Temporarily bind supported identifiers to all package log messages."""

    unknown = set(values) - set(_CONTEXT_NAMES)
    if unknown:
        joined = ", ".join(sorted(unknown))
        raise ValueError(f"unsupported logging context fields: {joined}")
    context = dict(_LOG_CONTEXT.get() or {})
    context.update({key: value for key, value in values.items() if value is not None})
    token = _LOG_CONTEXT.set(context)
    try:
        yield
    finally:
        _LOG_CONTEXT.reset(token)


def current_log_context() -> Mapping[str, str]:
    return dict(_LOG_CONTEXT.get() or {})
