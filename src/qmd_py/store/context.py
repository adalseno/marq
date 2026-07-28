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
from qmd_py.db.result import affected_rows
from qmd_py.store._common import _resolve_owned_collection


async def add_context(
    session: AsyncSession, user: CurrentUser, collection_name: str, path_prefix: str, text: str
) -> None:
    """Attach context prose to a collection, or to a path within it.

    Upserts: setting context for a prefix that already has one replaces
    the text rather than failing or accumulating.

    Contexts nest. A document inherits every context whose prefix it
    starts under, joined most-general-first, so a root context and a
    `journal/` context both reach `journal/entry.md`.

    Args:
        path_prefix: Path within the collection, or `""` for the whole
            collection. Matched as a path prefix, not a glob.
        text: The prose to attach. Search results carry it alongside the
            body so an agent knows what kind of document it is reading.

    Raises:
        CollectionNotFoundError: No collection of that name owned by `user`.
        PermissionDeniedError: `can_access()` refused `write` on it.
    """
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
    """Set the context that applies across all of this user's collections.

    The TS reference's `context add /`. User-scoped here, not a single
    system-wide value (see `User.global_context` in db/models.py). It is
    prepended before any per-path context, so it reads as the most general
    statement about everything this user has indexed.

    Args:
        text: The prose, or None to clear it.
    """
    result = await session.execute(select(User).where(col(User.id) == user.id))
    user_row = result.scalar_one()
    user_row.global_context = text
    session.add(user_row)
    await session.flush()


async def get_global_context(session: AsyncSession, user: CurrentUser) -> str | None:
    """Return this user's global context.

    Returns:
        The prose, or None when unset - which is also what a user row that
        somehow doesn't exist yields, since the caller has no use for the
        distinction.
    """
    result = await session.execute(select(col(User.global_context)).where(col(User.id) == user.id))
    return result.scalar_one_or_none()


@dataclass
class ContextRow:
    """One stored context entry, flattened for listing.

    Attributes:
        collection: Owning collection's name.
        path: The prefix this context applies to; `""` means the whole
            collection.
        context: The prose itself.
    """

    collection: str
    path: str
    context: str


async def list_contexts(session: AsyncSession, user: CurrentUser) -> list[ContextRow]:
    """Every context this user has set, across all their collections.

    Returns:
        Rows ordered by collection name, then path prefix - so a
        collection's root context sorts before its sub-path ones.

    Note:
        ACL (see auth.py): owner-prefiltered in SQL rather than gated
        through `can_access()`, so a granted-but-not-owned collection's
        contexts stay hidden until this query is widened.
    """
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
    """Delete the context set for one exact path prefix.

    Args:
        path_prefix: Must match what was stored exactly - this deletes one
            row, it does not clear a subtree.

    Returns:
        True if a row was deleted, False if there was nothing set for that
        prefix. Removing a context that isn't there is not an error.

    Raises:
        CollectionNotFoundError: No collection of that name owned by `user`.
        PermissionDeniedError: `can_access()` refused `write` on it.
    """
    collection = await _resolve_owned_collection(session, user, collection_name, "write")
    result = await session.execute(
        delete(CollectionContext).where(
            col(CollectionContext.collection_id) == collection.id,
            col(CollectionContext.path_prefix) == path_prefix,
        )
    )
    await session.flush()
    return affected_rows(result) > 0


@dataclass
class CollectionMissingContext:
    """A collection with no context set at all.

    Attributes:
        name: Collection name.
        path: Its indexed filesystem path.
        doc_count: Active documents in it - a rough measure of how much
            search quality is being left on the table.
    """

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
    TS inconsistencies).

    Two different gaps are reported, which is why the return value is a
    pair: a collection with *no* context at all, and a collection that has
    some context but leaves whole top-level directories uncovered. A
    collection with a root (`""`) context is never reported for the
    second, since that covers everything beneath it.

    Returns:
        A `(collections, paths)` pair. `collections` lists collections
        with no context, sorted by name. `paths` maps a collection name to
        its uncovered top-level directories, sorted, and omits collections
        with nothing missing - so an empty dict means fully covered.

    Note:
        ACL (see auth.py): owner-prefiltered in SQL rather than gated
        through `can_access()` - same widening needed as `list_contexts`
        when grants go live.

        Runs two to three queries per collection (an N+1), which is fine
        for an interactive advisory command.
    """
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
