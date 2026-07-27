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
    """The process-global async engine, built once from current settings.

    Cached deliberately - one connection pool per process - which is also
    why the test harness has to go through `reset_engine()` rather than
    passing an engine in. See its docstring.
    """
    return create_async_engine(get_settings().sqlalchemy_url)


@lru_cache
def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Session factory bound to `get_engine()`.

    `expire_on_commit=False` so objects stay usable after a commit -
    every service function returns ORM rows or dataclasses built from
    them, and re-fetching attributes post-commit would be unexpected I/O.
    """
    return async_sessionmaker(get_engine(), expire_on_commit=False)


@asynccontextmanager
async def get_session() -> AsyncIterator[AsyncSession]:
    """Open a session for one CLI command or MCP tool call.

    Does not commit: the caller decides, since a command may need several
    service calls in one transaction. The session closes on exit,
    rolling back anything uncommitted.
    """
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
