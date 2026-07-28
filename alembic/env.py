import asyncio
from logging.config import fileConfig
from typing import Literal

from sqlalchemy import pool, text
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel

from alembic import context
from qmd_py.config import get_settings

# Import models so their tables register on SQLModel.metadata before it's
# used below - this module is otherwise unused, hence the noqa.
from qmd_py.db import models  # noqa: F401

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = SQLModel.metadata

# sqlalchemy.url isn't set in alembic.ini - pulled directly from our own
# settings in run_async_migrations()/run_migrations_offline() below instead
# of via config.set_main_option(), which routes through configparser and
# chokes on the literal `%`s in a URL-encoded password/options string.
DB_URL = get_settings().sqlalchemy_url

# Isolation from the live TS reference implementation's tables (in `public`
# on the same `qmd` database) comes entirely from the connection's
# search_path being just `qmd_py` (see config.py) - none of our models
# declare an explicit `schema=`, so there's no multi-schema reflection to
# configure here at all. Two things still need filtering out:
# - Alembic's own `alembic_version` bookkeeping table: without this,
#   autogenerate proposes dropping and recreating the very table tracking
#   migration history, mid-migration.
# - `embeddings_<slug>` tables (see search/vector.py's ensure_embedding_model):
#   created dynamically at runtime, deliberately NOT part of SQLModel.metadata
#   (one physical table per embedding model, not knowable at migration-authoring
#   time) - without this, autogenerate proposes dropping every one it finds.
def include_name(
    name: str | None,
    type_: Literal[
        "schema", "table", "column", "index", "unique_constraint", "foreign_key_constraint"
    ],
    parent_names: object,
) -> bool:
    if type_ == "table":
        if name == "alembic_version":
            return False
        if name is not None and name.startswith("embeddings_"):
            return False
    elif isinstance(parent_names, dict):
        table_name = parent_names.get("table_name")
        if isinstance(table_name, str) and table_name.startswith("embeddings_"):
            return False
    return True


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    context.configure(
        url=DB_URL,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        include_name=include_name,
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        include_name=include_name,
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """In this scenario we need to create an Engine
    and associate a connection with the context.

    """

    connectable = create_async_engine(DB_URL, poolclass=pool.NullPool)

    async with connectable.connect() as connection:
        # Every table (and alembic_version itself) is created inside
        # MARQ_POSTGRES_SCHEMA, which nothing else creates: the connection
        # URL only sets it as the search_path, and Postgres will not create
        # a schema just because it is named there. On a blank database the
        # first `alembic upgrade head` therefore failed with "no schema has
        # been selected to create in" - which is exactly what the README
        # and quickstart tell a new user to run against a fresh container.
        # Idempotent, so it is a no-op on every subsequent run.
        await connection.execute(
            text(f'CREATE SCHEMA IF NOT EXISTS "{get_settings().postgres_schema}"')
        )
        await connection.commit()
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""

    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
