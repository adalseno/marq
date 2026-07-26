"""Schema-per-test-run Postgres harness.

Each test gets a fresh `qmd_test_<uuid>` schema on the real
`ubuserver.internal` instance (reusing its already-installed `vector`/
`pg_trgm` extensions rather than spinning up a throwaway containerized
Postgres), with tables created from `SQLModel.metadata` directly - not via
Alembic, since these tests care about behavior, not migration history.
The schema is dropped at teardown regardless of test outcome.
"""

import uuid
from collections.abc import AsyncIterator
from urllib.parse import quote

import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlmodel import SQLModel

from qmd_py.auth import CurrentUser, get_current_user
from qmd_py.config import get_settings


def _schema_url(schema: str) -> str:
    base_url = get_settings().postgres_url
    options = quote(f"-c search_path={schema}")
    separator = "&" if "?" in base_url else "?"
    return f"{base_url}{separator}options={options}"


@pytest_asyncio.fixture
async def engine() -> AsyncIterator[AsyncEngine]:
    schema = f"qmd_test_{uuid.uuid4().hex[:12]}"
    admin_engine = create_async_engine(get_settings().postgres_url)
    async with admin_engine.begin() as conn:
        await conn.execute(text(f'CREATE SCHEMA "{schema}"'))
    await admin_engine.dispose()

    test_engine = create_async_engine(_schema_url(schema))
    async with test_engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

    yield test_engine
    await test_engine.dispose()

    admin_engine = create_async_engine(get_settings().postgres_url)
    async with admin_engine.begin() as conn:
        await conn.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
    await admin_engine.dispose()


@pytest_asyncio.fixture
async def session(engine: AsyncEngine) -> AsyncIterator[AsyncSession]:
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with session_factory() as s:
        yield s


@pytest_asyncio.fixture
async def user(session: AsyncSession) -> CurrentUser:
    return await get_current_user(session)
