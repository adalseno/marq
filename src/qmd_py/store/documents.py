"""Content-addressable ingest primitives: the content/document row
writes every indexing path is built from. Deliberately low-level and
ACL-free - callers resolve the collection (and its permissions) first.
"""

from datetime import datetime

from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col

from qmd_py.db.models import Content, Document
from qmd_py.search.fts import update_document_search_vector


async def insert_content(session: AsyncSession, hash_: str, doc: str) -> None:
    stmt = (
        pg_insert(Content)
        .values(hash=hash_, doc=doc)
        .on_conflict_do_nothing(index_elements=["hash"])
    )
    await session.execute(stmt)


async def find_active_document(
    session: AsyncSession, collection_id: int, path: str
) -> Document | None:
    result = await session.execute(
        select(Document).where(
            col(Document.collection_id) == collection_id,
            col(Document.path) == path,
            col(Document.active),
        )
    )
    return result.scalar_one_or_none()


async def find_document_by_path(
    session: AsyncSession, collection_id: int, path: str
) -> Document | None:
    """Active or not - `reindex_collection` needs this (not
    `find_active_document`) to reactivate a deactivated document whose
    file reappears, rather than violating `Document`'s table-wide
    `UNIQUE(collection_id, path)` constraint by inserting a second row."""
    result = await session.execute(
        select(Document).where(
            col(Document.collection_id) == collection_id, col(Document.path) == path
        )
    )
    return result.scalar_one_or_none()


async def insert_document(
    session: AsyncSession,
    collection_id: int,
    path: str,
    title: str,
    hash_: str,
    created_at: datetime,
    modified_at: datetime,
) -> Document:
    document = Document(
        collection_id=collection_id,
        path=path,
        title=title,
        hash=hash_,
        created_at=created_at,
        modified_at=modified_at,
    )
    session.add(document)
    await session.flush()
    await update_document_search_vector(session, document.id)
    return document


async def update_document(
    session: AsyncSession, document: Document, title: str, hash_: str, modified_at: datetime
) -> None:
    """Also reactivates the document (`active=True`) unconditionally: every
    caller reaches this because a real file on disk currently maps to this
    (collection_id, path), including a file that reappeared after having
    been deactivated (e.g. switching git branches back and forth over the
    same indexed working tree) - `Document` has a table-wide
    `UNIQUE(collection_id, path)` constraint with no partial/active
    condition, so there can only ever be one row for a path regardless of
    active status; reactivating it is the only option, `INSERT`ing a
    second row isn't."""
    document.title = title
    document.hash = hash_
    document.modified_at = modified_at
    document.active = True
    session.add(document)
    await session.flush()
    await update_document_search_vector(session, document.id)


async def deactivate_document(session: AsyncSession, collection_id: int, path: str) -> None:
    await session.execute(
        update(Document)
        .where(col(Document.collection_id) == collection_id, col(Document.path) == path)
        .values(active=False)
    )
    await session.flush()


async def get_active_document_paths(session: AsyncSession, collection_id: int) -> list[str]:
    result = await session.execute(
        select(col(Document.path)).where(
            col(Document.collection_id) == collection_id, col(Document.active)
        )
    )
    return [p for (p,) in result.all()]
