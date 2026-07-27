"""Per-path and global context - the prose attached to a collection or
sub-path that search results carry alongside the document body.
"""

from dataclasses import dataclass

from sqlalchemy import delete, func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col

from qmd_py.auth import CurrentUser
from qmd_py.db.models import Collection, CollectionContext, Document, User
from qmd_py.store._common import _resolve_owned_collection


async def add_context(
    session: AsyncSession, user: CurrentUser, collection_name: str, path_prefix: str, text: str
) -> None:
    collection = await _resolve_owned_collection(session, user, collection_name, "write")
    stmt = (
        pg_insert(CollectionContext)
        .values(collection_id=collection.id, path_prefix=path_prefix, context=text)
        .on_conflict_do_update(
            index_elements=["collection_id", "path_prefix"], set_={"context": text}
        )
    )
    await session.execute(stmt)
    await session.flush()


async def set_global_context(session: AsyncSession, user: CurrentUser, text: str | None) -> None:
    """The TS reference's `context add /` - applies across all of a user's
    collections. User-scoped here, not a single system-wide value (see
    User.global_context's docstring in db/models.py)."""
    result = await session.execute(select(User).where(col(User.id) == user.id))
    user_row = result.scalar_one()
    user_row.global_context = text
    session.add(user_row)
    await session.flush()


async def get_global_context(session: AsyncSession, user: CurrentUser) -> str | None:
    result = await session.execute(select(col(User.global_context)).where(col(User.id) == user.id))
    return result.scalar_one_or_none()


@dataclass
class ContextRow:
    collection: str
    path: str
    context: str


async def list_contexts(session: AsyncSession, user: CurrentUser) -> list[ContextRow]:
    result = await session.execute(
        select(
            col(Collection.name), col(CollectionContext.path_prefix), col(CollectionContext.context)
        )
        .join(Collection, col(CollectionContext.collection_id) == Collection.id)
        .where(col(Collection.owner_user_id) == user.id)
        .order_by(col(Collection.name), col(CollectionContext.path_prefix))
    )
    return [ContextRow(collection=r[0], path=r[1], context=r[2]) for r in result.all()]


async def remove_context(
    session: AsyncSession, user: CurrentUser, collection_name: str, path_prefix: str
) -> bool:
    collection = await _resolve_owned_collection(session, user, collection_name, "write")
    result = await session.execute(
        delete(CollectionContext).where(
            col(CollectionContext.collection_id) == collection.id,
            col(CollectionContext.path_prefix) == path_prefix,
        )
    )
    await session.flush()
    return (result.rowcount or 0) > 0  # type: ignore[attr-defined]


@dataclass
class CollectionMissingContext:
    name: str
    path: str
    doc_count: int


async def context_check(
    session: AsyncSession, user: CurrentUser
) -> tuple[list[CollectionMissingContext], dict[str, list[str]]]:
    """Port of the TS reference's `getCollectionsWithoutContext` +
    `getTopLevelPathsWithoutContext`, combined into the one `context check`
    command they jointly back (which the TS CLI never actually wired up
    despite CLAUDE.md documenting it - see the qmd-py plan's list of fixed
    TS inconsistencies)."""
    collections_result = await session.execute(
        select(Collection).where(col(Collection.owner_user_id) == user.id)
    )
    collections = list(collections_result.scalars())

    missing_context: list[CollectionMissingContext] = []
    missing_paths: dict[str, list[str]] = {}

    for collection in collections:
        contexts_result = await session.execute(
            select(CollectionContext).where(col(CollectionContext.collection_id) == collection.id)
        )
        contexts = list(contexts_result.scalars())

        if not contexts:
            count_result = await session.execute(
                select(func.count(col(Document.id))).where(
                    col(Document.collection_id) == collection.id, col(Document.active)
                )
            )
            missing_context.append(
                CollectionMissingContext(
                    name=collection.name,
                    path=collection.path,
                    doc_count=count_result.scalar_one(),
                )
            )
            continue

        prefixes = {c.path_prefix for c in contexts}
        if "" in prefixes:
            continue  # root context covers the whole collection

        paths_result = await session.execute(
            select(col(Document.path)).where(
                col(Document.collection_id) == collection.id, col(Document.active)
            )
        )
        top_level_dirs = {
            parts[0]
            for (path,) in paths_result.all()
            if len(parts := [p for p in path.split("/") if p]) > 1
        }
        missing = sorted(
            d
            for d in top_level_dirs
            if not any(p == d or d.startswith(p + "/") for p in prefixes)
        )
        if missing:
            missing_paths[collection.name] = missing

    missing_context.sort(key=lambda c: c.name)
    return missing_context, missing_paths
