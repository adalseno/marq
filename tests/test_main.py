"""main()'s ValidationError -> friendly-message wrapper. Unlike the rest
of the CLI layer (see test_mcp.py's docstring), this is pure error-
formatting logic around a mocked failure, not a real command execution
against Postgres/the LLM router - so it's fair game for a direct unit
test rather than live-only verification.
"""

import logging
import sys

import pytest
from click.testing import CliRunner
from pydantic import ValidationError

from qmd_py.cli.main import cli, main
from qmd_py.config import Settings


def _missing_postgres_url_error(monkeypatch: pytest.MonkeyPatch) -> ValidationError:
    """Provoke the ValidationError main() is supposed to prettify.

    `_env_file=None` suppresses the `.env` file but *not* the process
    environment, so a real `MARQ_POSTGRES_URL` still satisfies the field
    and no error is raised. That made this helper pass locally (where the
    setting comes from `.env`) and fail under CI, which exports it as a
    genuine environment variable. Clear it explicitly instead of relying
    on the ambient environment.
    """
    monkeypatch.delenv("MARQ_POSTGRES_URL", raising=False)
    try:
        Settings(_env_file=None)
    except ValidationError as exc:
        return exc
    raise AssertionError("expected Settings() to raise without postgres_url")


def test_main_reports_missing_config_and_exits_nonzero(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    error = _missing_postgres_url_error(monkeypatch)

    def fake_cli() -> None:
        raise error

    monkeypatch.setattr("qmd_py.cli.main.cli", fake_cli)

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "invalid or missing configuration" in captured.err
    assert "MARQ_POSTGRES_URL" in captured.err


def test_main_lets_other_exceptions_propagate(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_cli() -> None:
        raise RuntimeError("something else entirely")

    monkeypatch.setattr("qmd_py.cli.main.cli", fake_cli)

    with pytest.raises(RuntimeError, match="something else entirely"):
        main()


@pytest.mark.parametrize(
    ("flags", "expected_level"),
    [([], logging.WARNING), (["-v"], logging.INFO), (["-vv"], logging.DEBUG)],
)
def test_verbose_flags_set_the_log_level(
    flags: list[str], expected_level: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """-v/-vv are the no-environment escape hatch for one-off debugging;
    without them the level comes from MARQ_LOG_LEVEL (WARNING by default)."""
    monkeypatch.setattr(logging.getLogger("qmd_py"), "handlers", [])
    # A real subcommand, not --help: click's --help is eager and exits
    # before the group callback (where logging is configured) ever runs.
    # `skills list` needs neither Postgres nor the router.
    CliRunner().invoke(cli, [*flags, "skills", "list"])

    assert logging.getLogger("qmd_py").level == expected_level


def test_cli_never_logs_to_stdout(monkeypatch: pytest.MonkeyPatch) -> None:
    """stdout carries parseable command output (--format json/csv), so a
    log line there would corrupt a pipeline."""
    monkeypatch.setattr(logging.getLogger("qmd_py"), "handlers", [])
    CliRunner().invoke(cli, ["-vv", "skills", "list"])

    for handler in logging.getLogger("qmd_py").handlers:
        assert getattr(handler, "stream", None) is not sys.stdout
