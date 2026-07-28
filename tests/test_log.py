"""Logging setup tests - `log.py`'s handler selection, level parsing and
idempotence, plus the two rules the module exists to enforce: nothing
ever goes to stdout, and the default level is silent on a healthy run.

Needs neither Postgres nor the router, so it runs in the default
`-m "not integration"` suite.
"""

import logging
from pathlib import Path

import pytest

from qmd_py.log import (
    LOG_FILE_BACKUP_COUNT,
    log_duration,
    request_context,
    setup_logging,
)


@pytest.fixture(autouse=True)
def _clean_logger() -> object:
    """Each test starts from a logger with no handlers, and leaves one -
    `setup_logging` is idempotent by design, so leftovers would make the
    next test's call a silent no-op."""
    logger = logging.getLogger("qmd_py")
    saved = list(logger.handlers)
    saved_level = logger.level
    logger.handlers.clear()
    yield None
    logger.handlers.clear()
    logger.handlers.extend(saved)
    logger.setLevel(saved_level)


def test_setup_logging_defaults_to_stderr_never_stdout(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The CLI's stdout is parseable output and the MCP stdio transport is
    JSON-RPC over stdout - a stray log line there corrupts both."""
    setup_logging("WARNING")
    logging.getLogger("qmd_py.test").warning("degraded thing happened")

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "degraded thing happened" in captured.err


def test_setup_logging_is_silent_below_the_configured_level(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The default must be silent on a healthy run: a non-empty log means
    something actually degraded."""
    setup_logging("WARNING")
    logging.getLogger("qmd_py.test").info("routine progress")

    assert capsys.readouterr().err == ""


def test_setup_logging_writes_to_a_rotating_file_when_asked(tmp_path: Path) -> None:
    log_file = tmp_path / "nested" / "marq.log"
    setup_logging("INFO", log_file)
    logging.getLogger("qmd_py.test").info("to the file")

    handler = logging.getLogger("qmd_py").handlers[0]
    assert isinstance(handler, logging.handlers.RotatingFileHandler)
    assert handler.backupCount == LOG_FILE_BACKUP_COUNT
    handler.flush()
    assert "to the file" in log_file.read_text()


def test_setup_logging_is_idempotent_but_force_replaces_the_handler(tmp_path: Path) -> None:
    """The CLI entry point and create_mcp_server() both call it; only the
    -v/-vv flags (force=True) should override what's already installed."""
    setup_logging("WARNING")
    setup_logging("DEBUG", tmp_path / "ignored.log")
    logger = logging.getLogger("qmd_py")
    assert len(logger.handlers) == 1
    assert isinstance(logger.handlers[0], logging.StreamHandler)
    assert logger.level == logging.WARNING

    setup_logging("DEBUG", force=True)
    assert len(logger.handlers) == 1
    assert logger.level == logging.DEBUG


def test_setup_logging_falls_back_to_warning_for_an_unknown_level() -> None:
    """Logging setup must never be the thing that breaks a command."""
    setup_logging("LOUD")
    assert logging.getLogger("qmd_py").level == logging.WARNING


def test_log_duration_reports_elapsed_time_and_caller_fields(
    caplog: pytest.LogCaptureFixture,
) -> None:
    setup_logging("INFO")
    with caplog.at_level(logging.INFO, logger="qmd_py.test"):
        with log_duration(logging.getLogger("qmd_py.test"), "query") as fields:
            fields["results"] = 9
            fields["candidates"] = 40

    assert "query: results=9 candidates=40" in caplog.text
    assert "ms" in caplog.text


def test_log_duration_still_logs_when_the_block_raises(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The timing of a call that blew up is the interesting one."""
    setup_logging("INFO")
    with caplog.at_level(logging.INFO, logger="qmd_py.test"), pytest.raises(RuntimeError):
        with log_duration(logging.getLogger("qmd_py.test"), "query"):
            raise RuntimeError("boom")

    assert "query:" in caplog.text


def test_request_context_tags_lines_and_restores_the_previous_id(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Concurrent MCP tool calls interleave their lines; the id is what
    makes one call's trace followable."""
    setup_logging("WARNING")
    logger = logging.getLogger("qmd_py.test")

    with request_context("query") as request_id:
        logger.warning("inside")
    logger.warning("outside")

    err = capsys.readouterr().err
    assert f"[{request_id}] inside" in err
    assert "[-] outside" in err
    assert request_id.startswith("query-")


def test_setup_logging_quiets_noisy_third_party_loggers() -> None:
    setup_logging("INFO")
    assert logging.getLogger("httpx").level == logging.WARNING
    assert logging.getLogger("sqlalchemy.engine").level == logging.WARNING

    setup_logging("DEBUG", force=True)
    assert logging.getLogger("httpx").level == logging.DEBUG
