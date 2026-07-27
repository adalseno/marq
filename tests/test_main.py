"""main()'s ValidationError -> friendly-message wrapper. Unlike the rest
of the CLI layer (see test_mcp.py's docstring), this is pure error-
formatting logic around a mocked failure, not a real command execution
against Postgres/the LLM router - so it's fair game for a direct unit
test rather than live-only verification.
"""

import pytest
from pydantic import ValidationError

from qmd_py.cli.main import main
from qmd_py.config import Settings


def _missing_postgres_url_error() -> ValidationError:
    try:
        Settings(_env_file=None)  # type: ignore[call-arg]
    except ValidationError as exc:
        return exc
    raise AssertionError("expected Settings() to raise without postgres_url")


def test_main_reports_missing_config_and_exits_nonzero(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    error = _missing_postgres_url_error()

    def fake_cli() -> None:
        raise error

    monkeypatch.setattr("qmd_py.cli.main.cli", fake_cli)

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "invalid or missing configuration" in captured.err
    assert "QMD_POSTGRES_URL" in captured.err


def test_main_lets_other_exceptions_propagate(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_cli() -> None:
        raise RuntimeError("something else entirely")

    monkeypatch.setattr("qmd_py.cli.main.cli", fake_cli)

    with pytest.raises(RuntimeError, match="something else entirely"):
        main()
