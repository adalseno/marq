"""`doctor` - diagnose Postgres/pgvector/migration/LLM-router health -
port of the TS reference's `showDoctor` (src/cli/qmd.ts), rescoped to
qmd-py's actual architecture: no local model files, GGUF inspection, GPU
device probing, or YAML config exist here, so those checks are dropped
rather than faithfully reproduced. The TS reference's embedding
"fingerprint" staleness checks are also dropped - they exist there to
detect drift in a single shared embeddings column; qmd-py's one-table-
per-model design (`embeddings_<slug>`) makes that failure mode
structurally impossible; a doc engine/pipeline change becomes a new
table, not silent drift in an existing one.
"""

import asyncio
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import click
import httpx
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from qmd_py.auth import get_current_user
from qmd_py.config import get_settings
from qmd_py.db.engine import get_engine, get_session
from qmd_py.llm.client import LlmClient
from qmd_py.search.vector import get_vector_index_health
from qmd_py.store import list_collections


def _check(label: str, ok: bool, details: str) -> None:
    mark = "✓" if ok else "⚠"
    click.echo(f"{mark} {label}: {details}")


def _redact_url(url: str) -> str:
    parts = urlsplit(url)
    if parts.password is None:
        return url
    netloc = parts.netloc.replace(f":{parts.password}", ":***")
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


def _find_alembic_ini() -> Path | None:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "alembic.ini"
        if candidate.exists():
            return candidate
    return None


async def _check_migrations(engine: AsyncEngine) -> tuple[bool, str]:
    ini_path = _find_alembic_ini()
    if ini_path is None:
        return False, "alembic.ini not found (doctor must run from the qmd-py source checkout)"

    from alembic.config import Config
    from alembic.runtime.migration import MigrationContext
    from alembic.script import ScriptDirectory

    cfg = Config(str(ini_path))
    script = ScriptDirectory.from_config(cfg)
    heads = set(script.get_heads())

    async with engine.connect() as conn:
        current = set(
            await conn.run_sync(
                lambda sync_conn: MigrationContext.configure(sync_conn).get_current_heads()
            )
        )

    if not current:
        return False, "no migrations applied yet. Next: `alembic upgrade head`"
    if current == heads:
        return True, f"up to date ({', '.join(sorted(current))})"
    return (
        False,
        f"pending migrations: at {', '.join(sorted(current))}, "
        f"head is {', '.join(sorted(heads))}. Next: `alembic upgrade head`",
    )


@click.command("doctor")
def doctor_command() -> None:
    """Diagnose Postgres, pgvector, migrations, and LLM-router health."""
    asyncio.run(_doctor_impl())


async def _doctor_impl() -> None:
    settings = get_settings()
    click.echo("QMD-py Doctor\n")
    click.echo(f"Postgres schema: {settings.postgres_schema}")
    click.echo(f"LLM router: {settings.llm_base_url}\n")

    engine = get_engine()

    async with get_session() as session:
        try:
            version = (await session.execute(text("SELECT version()"))).scalar_one()
            _check("Postgres connectivity", True, version)
        except Exception as exc:  # noqa: BLE001 - diagnostic, never fatal
            _check("Postgres connectivity", False, str(exc))

        try:
            row = (
                await session.execute(
                    text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
                )
            ).scalar_one_or_none()
            if row is not None:
                _check("pgvector extension", True, row)
            else:
                _check(
                    "pgvector extension", False,
                    "not installed; run `CREATE EXTENSION vector` as a superuser",
                )
        except Exception as exc:  # noqa: BLE001
            _check("pgvector extension", False, str(exc))

        try:
            ok, details = await _check_migrations(engine)
            _check("Migrations", ok, details)
        except Exception as exc:  # noqa: BLE001
            _check("Migrations", False, str(exc))

        user = await get_current_user(session)
        await session.commit()

        try:
            collections = await list_collections(session, user)
            if collections:
                _check("Collections", True, f"{len(collections)} configured")
            else:
                _check(
                    "Collections", False,
                    "none configured. Next: `qmdpy collection add .`",
                )
        except Exception as exc:  # noqa: BLE001
            _check("Collections", False, str(exc))

        try:
            health = await get_vector_index_health(session, user, settings.embed_model)
            if not health.has_vector_index:
                _check(
                    "Vector index", False,
                    f"no embeddings yet for {settings.embed_model}. Next: `qmdpy embed`",
                )
            elif health.needs_embedding > 0:
                _check(
                    "Vector index", False,
                    f"{health.needs_embedding} document(s) need embedding. Next: `qmdpy embed`",
                )
            else:
                _check("Vector index", True, f"{settings.embed_model} up to date")
        except Exception as exc:  # noqa: BLE001
            _check("Vector index", False, str(exc))

    try:
        async with LlmClient(settings.llm_base_url) as llm_client:
            models = await llm_client.list_models()
        _check("LLM router", True, f"reachable, {len(models)} model(s) available")
        for role, model in (
            ("embed", settings.embed_model),
            ("generate", settings.generate_model),
            ("rerank", settings.rerank_model),
        ):
            if model in models:
                _check(f"  {role} model", True, model)
            else:
                _check(f"  {role} model", False, f"{model} not found on router")
    except httpx.HTTPError as exc:
        _check("LLM router", False, f"unreachable: {exc}")

    click.echo("\nEffective configuration:")
    click.echo(f"  QMD_POSTGRES_URL       {_redact_url(settings.postgres_url)}")
    click.echo(f"  QMD_POSTGRES_SCHEMA    {settings.postgres_schema}")
    click.echo(f"  QMD_LLM_BASE_URL       {settings.llm_base_url}")
    click.echo(f"  QMD_EMBED_MODEL        {settings.embed_model}")
    click.echo(f"  QMD_GENERATE_MODEL     {settings.generate_model}")
    click.echo(f"  QMD_RERANK_MODEL       {settings.rerank_model}")
    click.echo(f"  QMD_DEFAULT_USER_EMAIL {settings.default_user_email}")
