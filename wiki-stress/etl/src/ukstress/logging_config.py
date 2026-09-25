"""Structured JSON logging for the ETL CLI.

Log records go to stderr as one JSON object per line, never to stdout:
every `ukstress` subcommand's stdout contract is a single final JSON result
object (see cli.py), and interleaving log lines into that stream would
break any script piping stdout through `json.loads`.

A record's `extra` fields are never a raw database URL or a raw SQL
exception object — see the individual call sites in database.py,
downloader.py, and pipeline.py for what each event actually carries.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any

_RESERVED_LOG_RECORD_FIELDS = frozenset(logging.makeLogRecord({}).__dict__)


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED_LOG_RECORD_FIELDS:
                payload[key] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)


def configure_logging(level: int = logging.INFO) -> None:
    """Install the JSON handler on the ukstress logger tree, idempotently."""
    logger = logging.getLogger("ukstress")
    logger.setLevel(level)
    logger.propagate = False
    if any(isinstance(handler, logging.StreamHandler) for handler in logger.handlers):
        return
    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(JSONFormatter())
    logger.addHandler(handler)
