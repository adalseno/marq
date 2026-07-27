"""Async SQLAlchemy engine/session factory.

Deliberately bypasses SQLModel's own (sync-oriented) `Session`/`create_engine`
helpers - SQLModel table classes are plain SQLAlchemy models, so they work
identically with SQLAlchemy's native async engine/session.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from functools import lru_cache

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from qmd_py.config import get_settings


@lru_cache
def get_engine() -> AsyncEngine:
    return create_async_engine(get_settings().sqlalchemy_url)


@lru_cache
def get_session_factory() -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(get_engine(), expire_on_commit=False)


@asynccontextmanager
async def get_session() -> AsyncIterator[AsyncSession]:
    async with get_session_factory()() as session:
        yield session


def reset_engine() -> None:
    """Drop the cached engine/session factory so the next `get_engine()`
    rebuilds from current settings.

    Exists for the test harness, which points `MARQ_POSTGRES_SCHEMA` at a
    throwaway schema and needs these process-global caches to notice (see
    tests/conftest.py). Also has to run between CLI invocations inside one
    process: every command body goes through `asyncio.run()`, which closes
    its loop on exit, and pooled asyncpg/psycopg connections do not
    survive their loop - reusing the cached engine across two
    `asyncio.run()` calls hands the second one connections bound to a dead
    loop.

    Discards the engine without `dispose()`ing it (that's async and this
    has to be callable from sync code); the dropped engine's pooled
    connections are closed when it is garbage collected.
    """
    get_session_factory.cache_clear()
    get_engine.cache_clear()
