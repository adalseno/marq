"""Phase 10: doctor command unit tests - the pure helpers (URL redaction,
alembic.ini discovery). The diagnostic checks themselves are exercised
live (`uv run qmdpy doctor` against the dev container and the real
server) rather than pytested, matching this project's established split
for CLI commands (see tests/test_mcp.py's module docstring)."""

from qmd_py.cli.commands.doctor import _find_alembic_ini, _redact_url


def test_redact_url_masks_password() -> None:
    redacted = _redact_url("postgresql+psycopg://user:secret@host/db")
    assert "secret" not in redacted
    assert redacted == "postgresql+psycopg://user:***@host/db"


def test_redact_url_no_password_is_unchanged() -> None:
    url = "postgresql+psycopg://host/db"
    assert _redact_url(url) == url


def test_find_alembic_ini_locates_project_root() -> None:
    ini_path = _find_alembic_ini()
    assert ini_path is not None
    assert ini_path.name == "alembic.ini"
    assert ini_path.is_file()
