"""Collection CRUD - `collection add/remove/rename/show/list` and the
include-by-default flag that decides what an unscoped query searches.

Named `collection` rather than `collections` so it can't be confused
with the stdlib module of that name.
"""

from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col

from qmd_py.auth import CurrentUser
from qmd_py.db.models import Collection, CollectionContext, CollectionGrant, Document
from qmd_py.store._common import _resolve_owned_collection, can_read
from qmd_py.store.cleanup import cleanup_orphaned_content


async def add_collection(
    session: AsyncSession,
    user: CurrentUser,
    name: str,
    path: str,
    pattern: str = "**/*.md",
    ignore: list[str] | None = None,
) -> Collection:
    collection = Collection(
        owner_user_id=user.id, name=name, path=path, pattern=pattern, ignore_patterns=ignore
    )
    session.add(collection)
    await session.flush()
    return collection


@dataclass
class RemoveCollectionResult:
    deleted_docs: int
    cleaned_hashes: int


async def remove_collection(
    session: AsyncSession, user: CurrentUser, name: str
) -> RemoveCollectionResult:
    """Deletes the collection's contexts and grants before the collection
    row itself - `collectioncontext_collection_id_fkey`/
    `collectiongrant_collection_id_fkey` have no `ON DELETE CASCADE`, so
    Postgres blocks the collection delete while either still references it
    (caught live: removing a collection with any per-path context still
    set raised a raw FK violation instead of succeeding)."""
    collection = await _resolve_owned_collection(session, user, name, "admin")
    deleted = await session.execute(
        delete(Document).where(col(Document.collection_id) == collection.id)
    )
    await session.execute(
        delete(CollectionContext).where(col(CollectionContext.collection_id) == collection.id)
    )
    await session.execute(
        delete(CollectionGrant).where(col(CollectionGrant.collection_id) == collection.id)
    )
    await session.delete(collection)
    await session.flush()
    cleaned = await cleanup_orphaned_content(session)
    return RemoveCollectionResult(
        deleted_docs=deleted.rowcount or 0,  # type: ignore[attr-defined]
        cleaned_hashes=cleaned,
    )


async def rename_collection(
    session: AsyncSession, user: CurrentUser, old_name: str, new_name: str
) -> None:
    collection = await _resolve_owned_collection(session, user, old_name, "admin")
    collection.name = new_name
    session.add(collection)
    await session.flush()


async def get_collection(session: AsyncSession, user: CurrentUser, name: str) -> Collection:
    return await _resolve_owned_collection(session, user, name, "read")


async def set_update_command(
    session: AsyncSession, user: CurrentUser, name: str, command: str | None
) -> None:
    collection = await _resolve_owned_collection(session, user, name, "admin")
    collection.update_command = command
    session.add(collection)
    await session.flush()


async def set_include_by_default(
    session: AsyncSession, user: CurrentUser, name: str, include: bool
) -> None:
    collection = await _resolve_owned_collection(session, user, name, "admin")
    collection.include_by_default = include
    session.add(collection)
    await session.flush()


@dataclass
class CollectionListRow:
    name: str
    path: str
    pattern: str
    doc_count: int
    active_count: int
    last_modified: datetime | None
    include_by_default: bool


async def list_collections(session: AsyncSession, user: CurrentUser) -> list[CollectionListRow]:
    """Only collections `user` can access (today: everything, since
    `can_access()` is mocked True - but the filter is real, not skipped)."""
    result = await session.execute(select(Collection).order_by(col(Collection.name)))
    rows = []
    for collection in result.scalars():
        if not await can_read(user, collection):
            continue
        stats = await session.execute(
            select(
                func.count(col(Document.id)),
                func.max(col(Document.modified_at)),
            ).where(col(Document.collection_id) == collection.id, col(Document.active))
        )
        active_count, last_modified = stats.one()
        rows.append(
            CollectionListRow(
                name=collection.name,
                path=collection.path,
                pattern=collection.pattern,
                doc_count=active_count or 0,
                active_count=active_count or 0,
                last_modified=last_modified,
                include_by_default=collection.include_by_default,
            )
        )
    return rows
