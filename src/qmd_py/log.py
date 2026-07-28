"""Logging setup for the `qmd_py` logger hierarchy.

Two hard rules, both specific to this project:

**Stdout is never a log destination.** The CLI's stdout is parseable
output (`--format json/csv` pipelines), and the MCP stdio transport *is*
JSON-RPC over stdout - a single stray log line corrupts the protocol.
Logging goes to stderr or a file, period.

**Log shapes, not content.** At WARNING and INFO, lines carry counts,
lengths, paths, durations, and exception types - never queries, document
bodies, or snippets. The index holds whatever the user pointed it at; a
log file that quietly re-hosts fragments of it is a leak vector,
especially once the log outlives the collection (`marq collection
remove` deletes documents, not log lines). Content is allowed at DEBUG
only, which is opt-in.

The default level (WARNING) is silent on a healthy run - that's the
discipline that keeps the log trustworthy: a non-empty log means
something actually degraded. Every module logs through the standard
`logging.getLogger(__name__)`, so the `qmd_py.search` /
`qmd_py.store` hierarchy comes for free and one subsystem can be turned
up later without a firehose from the rest.

Stdlib logging on purpose (no structlog/loguru): there is no aggregation
stack to feed, plain formatted lines are grep-able and diff-able,
pytest's `caplog` works out of the box, and a JSON formatter can be
swapped into the same handler later if a collector ever appears.
"""

import contextvars
import logging
import logging.handlers
import secrets
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s [%(request_id)s] %(message)s"

_VALID_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})

_NOISY_LOGGERS = ("httpx", "httpcore", "sqlalchemy.engine", "mcp", "uvicorn")
"""Third-party libraries whose INFO/DEBUG output would drown ours. They
stay at WARNING unless the whole setup is explicitly at DEBUG - only
then is `sqlalchemy.engine`'s SQL echo and httpx's per-request line
wanted."""

LOG_FILE_MAX_BYTES = 5_000_000
LOG_FILE_BACKUP_COUNT = 3


# =============================================================================
# Request correlation
# =============================================================================

_request_id: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")
"""MCP tool calls and REST requests are concurrent, so their log lines
interleave. A short id per request, injected into every line through
`_RequestIdFilter`, is the difference between a usable and a useless
daemon log. A contextvar (not a thread-local) because the server is
async: each task inherits its own copy."""


class _RequestIdFilter(logging.Filter):
    """Makes `%(request_id)s` available to the formatter on every record,
    whether or not the emitting code knows about request ids at all."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = _request_id.get()
        return True


@contextmanager
def request_context(prefix: str = "") -> Iterator[str]:
    """Tag every log line emitted inside this block with a fresh id.

    Args:
        prefix: Short label for the entry point (e.g. `"query"`,
            `"rest"`), so an id is readable as well as unique.

    Yields:
        The generated id, in case the caller wants to return it to the
        client for cross-referencing.
    """
    request_id = f"{prefix}{'-' if prefix else ''}{secrets.token_hex(3)}"
    token = _request_id.set(request_id)
    try:
        yield request_id
    finally:
        _request_id.reset(token)


@contextmanager
def log_duration(
    logger: logging.Logger, label: str, level: int = logging.INFO
) -> Iterator[dict[str, object]]:
    """Time a block and log `label` with its elapsed milliseconds.

    Keeps the INFO timing story from cluttering the pipeline it measures.
    The yielded dict is merged into the logged line, so a caller can
    attach counts (never content) gathered inside the block:

        with log_duration(logger, "query") as fields:
            ...
            fields["results"] = len(results)
    """
    fields: dict[str, object] = {}
    started = time.perf_counter()
    try:
        yield fields
    finally:
        elapsed_ms = (time.perf_counter() - started) * 1000
        if logger.isEnabledFor(level):
            extra = " ".join(f"{k}={v}" for k, v in fields.items())
            logger.log(level, "%s: %s%s%.0fms", label, extra, " " if extra else "", elapsed_ms)


def setup_logging(level: str, log_file: Path | None = None, *, force: bool = False) -> None:
    """Configure the `qmd_py` logger with one handler: a size-rotated file
    when `log_file` is given, stderr otherwise.

    Idempotent: a second call is a no-op while a handler is installed,
    so the CLI entry point and `create_mcp_server()` can both call it
    without double-logging. `force=True` replaces the existing handler -
    the `-v/--verbose` flags use it to override the environment's level.

    An unknown `level` falls back to WARNING rather than raising:
    logging setup must never be the thing that breaks a command.
    """
    root = logging.getLogger("qmd_py")
    if root.handlers and not force:
        return
    for existing in list(root.handlers):
        root.removeHandler(existing)

    normalized = level.upper()
    if normalized not in _VALID_LEVELS:
        normalized = "WARNING"
    root.setLevel(normalized)

    handler: logging.Handler
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=LOG_FILE_MAX_BYTES, backupCount=LOG_FILE_BACKUP_COUNT
        )
    else:
        handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(_FORMAT))
    # On the handler, not the logger: a filter on a logger doesn't run for
    # records propagating up from its children, and every real call site
    # is a child (`qmd_py.search.hybrid`, `qmd_py.store.indexing`, ...).
    handler.addFilter(_RequestIdFilter())
    root.addHandler(handler)

    for noisy in _NOISY_LOGGERS:
        logging.getLogger(noisy).setLevel(
            logging.DEBUG if normalized == "DEBUG" else logging.WARNING
        )
