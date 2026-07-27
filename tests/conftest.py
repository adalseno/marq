"""Schema-per-test-run Postgres harness.

Each test gets a fresh `qmd_test_<uuid>` schema on the real
`ubuserver.internal` instance (reusing its already-installed `vector`/
`pg_trgm` extensions rather than spinning up a throwaway containerized
Postgres), with tables created from `SQLModel.metadata` directly - not via
Alembic, since these tests care about behavior, not migration history.
The schema is dropped at teardown regardless of test outcome.
"""

import asyncio
import uuid
from collections.abc import AsyncIterator, Callable, Iterator
from pathlib import Path
from urllib.parse import quote

import pytest
import pytest_asyncio
from click.testing import CliRunner, Result
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlmodel import SQLModel

from qmd_py.auth import CurrentUser, get_current_user
from qmd_py.cli.main import cli as cli_group
from qmd_py.config import get_settings
from qmd_py.db.engine import get_engine, reset_engine
from qmd_py.db.models import Collection
from qmd_py.store import (
    add_collection,
    extract_title,
    hash_content,
    insert_content,
    insert_document,
    utcnow,
)

FIXTURE_COLLECTION_PATH = Path(__file__).parent / "fixtures" / "sample-collection"
FIXTURE_COLLECTION_EXTS = {".md", ".py", ".js", ".ts"}


def _schema_url(schema: str) -> str:
    # Single-entry search_path, deliberately - see config.py's
    # sqlalchemy_url for why a multi-entry search_path is dangerous here:
    # Postgres's "is this object visible" check (used by both
    # pg_table_is_visible() for Alembic and CREATE TABLE IF NOT EXISTS)
    # considers a name visible if ANY search_path entry resolves to it. A
    # first attempt at fixing pgvector's `vector` type visibility (below)
    # by adding the app's `qmd_py` schema to this search_path caused every
    # ordinary SQLModel table's `CREATE TABLE IF NOT EXISTS` to silently
    # no-op (already visible in qmd_py) and all test data to leak into the
    # live app schema instead of this isolated one - see
    # search/vector.py's `_pgvector_schema` for the real, narrow fix.
    base_url = get_settings().postgres_url
    options = quote(f"-c search_path={schema}")
    separator = "&" if "?" in base_url else "?"
    return f"{base_url}{separator}options={options}"


def _new_schema_name() -> str:
    return f"qmd_test_{uuid.uuid4().hex[:12]}"


async def _create_test_schema(schema: str) -> None:
    admin_engine = create_async_engine(get_settings().postgres_url)
    async with admin_engine.begin() as conn:
        await conn.execute(text(f'CREATE SCHEMA "{schema}"'))
    await admin_engine.dispose()

    setup_engine = create_async_engine(_schema_url(schema))
    async with setup_engine.begin() as conn:
        # Extension objects (e.g. pgvector's `vector` type) live in whatever
        # schema was on the search_path when CREATE EXTENSION ran - the
        # `qmd_py` schema's copy isn't visible from this fresh test schema.
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))
        await conn.run_sync(SQLModel.metadata.create_all)
    await setup_engine.dispose()


async def _drop_test_schema(schema: str) -> None:
    admin_engine = create_async_engine(get_settings().postgres_url)
    async with admin_engine.begin() as conn:
        await conn.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
    await admin_engine.dispose()


@pytest_asyncio.fixture
async def engine() -> AsyncIterator[AsyncEngine]:
    schema = _new_schema_name()
    await _create_test_schema(schema)
    test_engine = create_async_engine(_schema_url(schema))
    yield test_engine
    await test_engine.dispose()
    await _drop_test_schema(schema)


@pytest_asyncio.fixture
async def session(engine: AsyncEngine) -> AsyncIterator[AsyncSession]:
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with session_factory() as s:
        yield s


@pytest_asyncio.fixture
async def user(session: AsyncSession) -> CurrentUser:
    return await get_current_user(session)


@pytest_asyncio.fixture
async def sample_collection(session: AsyncSession, user: CurrentUser) -> Collection:
    """Indexes `tests/fixtures/sample-collection/` - a small, frozen,
    checked-in multi-language project (.md/.py/.js/.ts) - into a real
    collection. A manageable, always-available stand-in for the external
    book-catalog fixture used during development (which isn't checked into
    the repo and needs SSH access to reindex), so anyone can run realistic
    multi-file FTS/vector search tests without either."""
    collection = await add_collection(
        session, user, "sample", str(FIXTURE_COLLECTION_PATH), "**/*.{md,py,js,ts}"
    )
    for filepath in sorted(FIXTURE_COLLECTION_PATH.rglob("*")):
        if not filepath.is_file() or filepath.suffix not in FIXTURE_COLLECTION_EXTS:
            continue
        rel_path = str(filepath.relative_to(FIXTURE_COLLECTION_PATH))
        content = filepath.read_text(encoding="utf-8")
        digest = await hash_content(content)
        title = extract_title(content, rel_path)
        await insert_content(session, digest, content)
        await insert_document(session, collection.id, rel_path, title, digest, utcnow(), utcnow())
    await session.commit()
    return collection


# =============================================================================
# Process-global-engine harness (CLI / MCP)
#
# The fixtures above hand tests their own `AsyncEngine`. The CLI and MCP
# server can't take one: both reach `db/engine.py`'s process-global,
# `lru_cache`'d `get_engine()`, resolved from `MARQ_*` settings that are
# themselves `lru_cache`'d. The two fixtures below point that global at a
# throwaway schema instead - patch `MARQ_POSTGRES_SCHEMA`, drop both caches,
# and drop them again on the way out so nothing leaks into the next test.
# =============================================================================


@pytest.fixture
def marq(monkeypatch: pytest.MonkeyPatch) -> Iterator[Callable[..., Result]]:
    """Invokes real `marq` CLI commands against a throwaway schema.

    Yields a callable taking argv as separate strings - `marq("ls",
    "docs")` - and returning click's `Result` (`.exit_code`, `.output`).

    Deliberately sync, and so are its tests: every command body ends in
    `cli/runtime.py`'s `asyncio.run()`, which raises outright if a loop is
    already running. `reset_engine()` before each invocation is what makes
    a second command in the same test work at all - see its docstring.
    """
    schema = _new_schema_name()
    asyncio.run(_create_test_schema(schema))
    monkeypatch.setenv("MARQ_POSTGRES_SCHEMA", schema)
    get_settings.cache_clear()
    reset_engine()

    def invoke(*args: str) -> Result:
        reset_engine()
        return CliRunner().invoke(cli_group, list(args))

    yield invoke

    # Undo explicitly rather than leaving it to monkeypatch's own teardown:
    # the settings cache has to be cleared *after* the env var is gone, or
    # the next test that reads settings gets this dead schema back.
    monkeypatch.undo()
    reset_engine()
    get_settings.cache_clear()
    asyncio.run(_drop_test_schema(schema))


@pytest_asyncio.fixture
async def mcp_env(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[str]:
    """Same global-engine redirection as `marq`, for the MCP server's async
    tool handlers (which open their own `get_session()` per call).

    Async here, unlike `marq`: MCP handlers are awaited directly, so there
    is no nested-`asyncio.run()` problem and the global engine can bind to
    the test's own running loop. Yields the schema name.
    """
    schema = _new_schema_name()
    await _create_test_schema(schema)
    monkeypatch.setenv("MARQ_POSTGRES_SCHEMA", schema)
    get_settings.cache_clear()
    reset_engine()

    yield schema

    # Dispose inside the test's loop; the pooled connections don't outlive it.
    await get_engine().dispose()
    monkeypatch.undo()
    reset_engine()
    get_settings.cache_clear()
    await _drop_test_schema(schema)
