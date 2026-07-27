"""Phase 10/11: doctor command unit tests - the pure helpers (URL
redaction, alembic.ini discovery) plus _check_migrations()'s "no
migrations applied" branch, which is directly testable against the
`engine` fixture's schema (it never runs Alembic - see conftest.py -
so it always looks alembic-unmigrated). The "up to date"/"pending
migrations" branches need a schema Alembic actually migrated, which
isn't this fixture's job; those stay covered by live verification
(`uv run marq doctor` / `uv run alembic check` against the dev
container and the real server) rather than pytested, matching this
project's established split for CLI commands (see tests/test_mcp.py's
module docstring)."""

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine

from qmd_py.cli.commands.doctor import _check_migrations, _find_alembic_ini, _redact_url


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


@pytest.mark.integration
async def test_check_migrations_reports_none_applied_on_fresh_schema(engine: AsyncEngine) -> None:
    ok, details = await _check_migrations(engine)
    assert ok is False
    assert "no migrations applied yet" in details
