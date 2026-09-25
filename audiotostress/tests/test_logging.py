import io
import json
import logging

import pytest

from ukstress.provenance.logging import bind_log_context, configure_logging


def test_logging_includes_bound_pipeline_context() -> None:
    stream = io.StringIO()
    configure_logging(stream=stream)

    with bind_log_context(run_id="run-1", shard_id="007"), bind_log_context(
        source_id="commonvoice", record_id="rec-1"
    ):
        logging.getLogger("ukstress.pipeline").info("accepted %s", "замок")

    payload = json.loads(stream.getvalue())
    assert payload["message"] == "accepted замок"
    assert payload["run_id"] == "run-1"
    assert payload["shard_id"] == "007"
    assert payload["source_id"] == "commonvoice"
    assert payload["record_id"] == "rec-1"


def test_logging_context_is_reset_after_scope() -> None:
    stream = io.StringIO()
    configure_logging(stream=stream)
    logger = logging.getLogger("ukstress.test")

    with bind_log_context(run_id="run-1"):
        logger.info("inside")
    logger.info("outside")

    first, second = (json.loads(line) for line in stream.getvalue().splitlines())
    assert first["run_id"] == "run-1"
    assert "run_id" not in second


def test_logging_rejects_unknown_context() -> None:
    with pytest.raises(ValueError, match="unsupported"), bind_log_context(
        speaker_id="private"  # type: ignore[arg-type]
    ):
        pass
