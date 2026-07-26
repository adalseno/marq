"""SQLModel table definitions.

Table classes here are plain SQLAlchemy models underneath (SQLModel is a thin
Pydantic layer on top) - all async DB access goes through SQLAlchemy's own
AsyncEngine/AsyncSession (see engine.py), never SQLModel's sync-oriented
`Session`/`create_engine` helpers. Columns SQLModel's type system doesn't
know natively (`TSVECTOR`, `pgvector.Vector`) use the `sa_column=` escape
hatch, same as plain SQLAlchemy would need.

Note on `search_vector`: this is a plain, app-maintained TSVECTOR column,
NOT a Postgres `GENERATED ALWAYS AS (...) STORED` column - a generated
column's expression can only reference columns in its own table, but the
document body lives in `Content.doc` (joined via `hash`), not in
`Document` itself. It's kept in sync by `update_document_search_vector()`
in the search module (ported from the TS `updateDocumentSearchVector`),
called wherever a document's title/path/body changes - the same discipline
the working TS/Postgres implementation already uses successfully.

Per-model embedding tables (`embeddings_<slug>`) are NOT defined here -
they're created dynamically at runtime once a model's dimension is known
(see search/vector.py, Phase 5), mirroring the TS version's lazy vector
column creation but non-destructively (one table per model, never dropped
on a model switch).
"""

from datetime import datetime
from typing import Any

from sqlalchemy import (
    ARRAY,
    TIMESTAMP,
    BigInteger,
    CheckConstraint,
    Column,
    ForeignKey,
    Identity,
    Index,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR
from sqlmodel import Field, SQLModel

# No explicit `schema=` anywhere below, deliberately: qmd-py's isolation
# from the TS/Postgres reference implementation's tables (which live in
# `public` on the same `qmd` database - see the plan's "Single Postgres,
# forever" decision) comes entirely from the connection's `search_path`
# (set to just `qmd_py` in config.py), the same way the TS implementation
# itself creates unqualified tables that land whichever schema its own
# connection defaults to.
#
# An earlier version of this file set `SQLModel.metadata.schema = "qmd_py"`
# explicitly and schema-qualified every FK target string. That produced a
# permanent, recurring false-positive `alembic check`/autogenerate diff:
# Alembic's reflection of the connection's *default* schema normalizes
# table/FK objects to `schema=None` (since from the connection's point of
# view, unqualified access already means "the current schema"), while an
# explicit `schema="qmd_py"` on the metadata side is a *different* Python
# representation of the exact same schema - and Alembic's constraint
# comparator does a structural comparison that doesn't collapse the two,
# so it proposed dropping and recreating the same FKs forever. Leaving
# schema unset here makes both sides agree (`schema=None` on both), which
# is also simply less code.


class User(SQLModel, table=True):
    """Exactly one row for now (the mocked single-user case) - see auth.py.

    `global_context` is the TS version's `context add /` ("applies to all
    collections") - user-scoped here rather than a single system-wide KV
    row, since in a real multi-user future each user reasonably wants their
    own global context, not one shared by everyone."""

    id: int = Field(sa_column=Column(BigInteger, Identity(), primary_key=True))
    email: str = Field(unique=True, index=True)
    display_name: str | None = None
    is_admin: bool = False
    global_context: str | None = None
    created_at: datetime = Field(
        sa_column=Column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)
    )


class Collection(SQLModel, table=True):
    """Replaces the TS version's bare `documents.collection` TEXT tag with a
    real, owned entity - the prerequisite for per-collection ACL. `name` is
    scoped per-owner (UNIQUE(owner_user_id, name)), not globally unique like
    the TS version's `store_collections.name` primary key.

    Per-path context text (the TS version's `context add <path> "text"`,
    where `<path>` can be the collection root or a sub-path) lives in
    `CollectionContext`, not as a single string field here - the TS model
    supports many contexts per collection, keyed by path prefix."""

    __table_args__ = (UniqueConstraint("owner_user_id", "name"),)

    id: int = Field(sa_column=Column(BigInteger, Identity(), primary_key=True))
    owner_user_id: int = Field(
        sa_column=Column(BigInteger, ForeignKey("user.id"), index=True, nullable=False)
    )
    name: str
    path: str
    pattern: str = "**/*.md"
    ignore_patterns: list[str] | None = Field(default=None, sa_column=Column(ARRAY(Text)))
    include_by_default: bool = True
    update_command: str | None = None
    created_at: datetime = Field(
        sa_column=Column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)
    )
    updated_at: datetime = Field(
        sa_column=Column(
            TIMESTAMP(timezone=True),
            server_default=func.now(),
            onupdate=func.now(),
            nullable=False,
        )
    )


class CollectionContext(SQLModel, table=True):
    """One row per (collection, path_prefix) context entry - `path_prefix`
    is `""` for the collection root (`context add qmd://coll/` with no
    sub-path), or a relative sub-path (`context add qmd://coll/notes`)."""

    __table_args__ = (UniqueConstraint("collection_id", "path_prefix"),)

    id: int = Field(sa_column=Column(BigInteger, Identity(), primary_key=True))
    collection_id: int = Field(
        sa_column=Column(BigInteger, ForeignKey("collection.id"), index=True, nullable=False)
    )
    path_prefix: str = ""
    context: str = Field(sa_column=Column(Text, nullable=False))
    created_at: datetime = Field(
        sa_column=Column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)
    )


class CollectionGrant(SQLModel, table=True):
    """Additive ACL table - unpopulated beyond the implicit owner today.
    `can_access()` in auth.py is mocked to always return True regardless of
    whether a grant row exists; wiring this table into real checks later is
    additive, not a rearchitecture."""

    __table_args__ = (
        UniqueConstraint("collection_id", "user_id"),
        CheckConstraint("permission IN ('read', 'write', 'admin')", name="ck_grant_permission"),
    )

    id: int = Field(sa_column=Column(BigInteger, Identity(), primary_key=True))
    collection_id: int = Field(
        sa_column=Column(
            BigInteger, ForeignKey("collection.id"), index=True, nullable=False
        )
    )
    user_id: int = Field(
        sa_column=Column(BigInteger, ForeignKey("user.id"), index=True, nullable=False)
    )
    permission: str
    granted_at: datetime = Field(
        sa_column=Column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)
    )


class Content(SQLModel, table=True):
    """Content-addressable storage - unchanged from the TS design (it's
    good, not legacy): identical document bodies dedupe across paths and
    collections by hash."""

    hash: str = Field(sa_column=Column(Text, primary_key=True))
    doc: str
    created_at: datetime = Field(
        sa_column=Column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)
    )


class Document(SQLModel, table=True):
    """`collection_id` FK replaces the TS version's bare TEXT tag;
    `created_at`/`modified_at` are real TIMESTAMPTZ (not TS's TEXT ISO
    strings); `active` is a real BOOLEAN (not TS's INTEGER 0/1)."""

    __table_args__ = (
        UniqueConstraint("collection_id", "path"),
        Index("ix_document_search_vector", "search_vector", postgresql_using="gin"),
    )

    id: int = Field(sa_column=Column(BigInteger, Identity(), primary_key=True))
    collection_id: int = Field(
        sa_column=Column(
            BigInteger, ForeignKey("collection.id"), index=True, nullable=False
        )
    )
    path: str
    title: str
    hash: str = Field(sa_column=Column(Text, ForeignKey("content.hash"), nullable=False))
    created_at: datetime = Field(sa_column=Column(TIMESTAMP(timezone=True), nullable=False))
    modified_at: datetime = Field(sa_column=Column(TIMESTAMP(timezone=True), nullable=False))
    active: bool = True
    search_vector: str | None = Field(default=None, sa_column=Column(TSVECTOR, nullable=True))


class LlmCache(SQLModel, table=True):
    """`result` is JSONB (not TS's TEXT) - queryable, and the natural type
    for what's always JSON-serialized data anyway."""

    hash: str = Field(primary_key=True)
    result: dict[str, Any] = Field(sa_column=Column(JSONB, nullable=False))
    created_at: datetime = Field(
        sa_column=Column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)
    )


class EmbeddingModel(SQLModel, table=True):
    """Registry of embedding models that have ever been used - each active
    row corresponds to one dynamically-created `embeddings_<slug>` table
    (see search/vector.py, Phase 5), fixing the TS version's one-model-at-
    a-time limitation."""

    __tablename__ = "embedding_model"

    id: int = Field(sa_column=Column(BigInteger, Identity(), primary_key=True))
    slug: str = Field(unique=True, index=True)
    role: str = Field(sa_column=Column(Text, nullable=False))
    """'embed' | 'rerank' | 'generate' - CHECK constraint below, not an enum type."""
    dimension: int | None = None
    is_default: bool = False

    __table_args__ = (
        CheckConstraint("role IN ('embed', 'rerank', 'generate')", name="ck_embedding_model_role"),
    )
