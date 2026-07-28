"""Collection CRUD - `collection add/remove/rename/show/list` and the
include-by-default flag that decides what an unscoped query searches.

Named `collection` rather than `collections` so it can't be confused
with the stdlib module of that name.

Docstrings here are Google style (`docstring_style = "google"` in
zensical.toml) and serve as this project's worked reference for the
format. Two conventions worth copying:

- **No types in docstrings.** Every parameter is annotated and the code is
  `mypy --strict`, so mkdocstrings renders the real signature
  (`show_signature_annotations`). Repeating `name (str)` duplicates the
  annotation somewhere mypy can't check, and it drifts.
- **`Args:` is optional, `Raises:` usually isn't.** `session`/`user` repeat
  on nearly every function here and documenting them each time is noise;
  which exception a caller gets, and when, is genuinely non-obvious.

Every function below leaves the transaction open - they `flush()` so
generated ids and rowcounts are available, and the caller commits.
"""

from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col

from qmd_py.auth import CurrentUser
from qmd_py.db.models import Collection, CollectionContext, CollectionGrant, Document
from qmd_py.db.result import affected_rows
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
    """Register a new collection. Does not index it.

    Indexing is a separate step: call `reindex_collection()` afterwards
    (that is what `marq collection add` does), so a caller can create and
    populate a collection in one transaction.

    Args:
        name: Unique per owner, not globally - two users may each have a
            collection called `notes`.
        path: Absolute filesystem path to walk. Not validated here; a
            path that doesn't exist simply yields no files at reindex.
        pattern: Glob relative to `path`. Brace groups are expanded
            (`{src,docs}/**/*.{md,py}`), which plain `glob` cannot do.
        ignore: Extra glob patterns to skip, on top of the always-excluded
            hidden files and `node_modules`/`.git`/`dist`-style directories.

    Returns:
        The persisted row, with `id` populated by the flush.

    Raises:
        sqlalchemy.exc.IntegrityError: This owner already has a collection
            of that name (`UNIQUE(owner_user_id, name)`). Raised on flush,
            so the caller must roll back before reusing the session.
    """
    collection = Collection(
        owner_user_id=user.id, name=name, path=path, pattern=pattern, ignore_patterns=ignore
    )
    session.add(collection)
    await session.flush()
    return collection


@dataclass
class RemoveCollectionResult:
    """What `remove_collection()` reclaimed.

    Attributes:
        deleted_docs: Document rows removed, active and inactive alike.
        cleaned_hashes: Content rows dropped because no document anywhere
            still referenced them. Lower than `deleted_docs` whenever
            another collection shares the same file content, since content
            is addressed by hash and shared across collections.
    """

    deleted_docs: int
    cleaned_hashes: int


async def remove_collection(
    session: AsyncSession, user: CurrentUser, name: str
) -> RemoveCollectionResult:
    """Delete a collection, its documents, contexts and grants.

    Deletes the contexts and grants *before* the collection row -
    `collectioncontext_collection_id_fkey`/
    `collectiongrant_collection_id_fkey` have no `ON DELETE CASCADE`, so
    Postgres blocks the collection delete while either still references it
    (caught live: removing a collection with any per-path context still
    set raised a raw FK violation instead of succeeding).

    Returns:
        Counts of what was removed; see `RemoveCollectionResult`.

    Raises:
        CollectionNotFoundError: No collection of that name owned by
            `user`. Note this is also what a *non-owner* gets, because the
            lookup prefilters on ownership in SQL - see auth.py.
        PermissionDeniedError: `can_access()` refused `admin` on it.
    """
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
        deleted_docs=affected_rows(deleted),
        cleaned_hashes=cleaned,
    )


async def rename_collection(
    session: AsyncSession, user: CurrentUser, old_name: str, new_name: str
) -> None:
    """Rename a collection in place, keeping its documents and contexts.

    Raises:
        CollectionNotFoundError: No collection named `old_name` owned by
            `user`.
        PermissionDeniedError: `can_access()` refused `admin` on it.
        sqlalchemy.exc.IntegrityError: `new_name` is already taken by
            another of this owner's collections. Raised on flush, not
            checked up front.
    """
    collection = await _resolve_owned_collection(session, user, old_name, "admin")
    collection.name = new_name
    session.add(collection)
    await session.flush()


async def get_collection(session: AsyncSession, user: CurrentUser, name: str) -> Collection:
    """Fetch one collection by name, gated for read access.

    Returns:
        The ORM row, so callers can read `path`/`pattern`/`update_command`
        directly.

    Raises:
        CollectionNotFoundError: No collection of that name owned by `user`.
        PermissionDeniedError: `can_access()` refused `read` on it.
    """
    return await _resolve_owned_collection(session, user, name, "read")


async def set_update_command(
    session: AsyncSession, user: CurrentUser, name: str, command: str | None
) -> None:
    """Set the shell command run before this collection is re-indexed.

    Used for collections backed by something that has to be refreshed
    first - a `git pull`, an export script - so `marq update` can bring
    the source up to date before walking it.

    Args:
        command: Shell command to run from the collection's path, or None
            to clear it.

    Raises:
        CollectionNotFoundError: No collection of that name owned by `user`.
        PermissionDeniedError: `can_access()` refused `admin` on it.
    """
    collection = await _resolve_owned_collection(session, user, name, "admin")
    collection.update_command = command
    session.add(collection)
    await session.flush()


async def set_include_by_default(
    session: AsyncSession, user: CurrentUser, name: str, include: bool
) -> None:
    """Include or exclude this collection from unscoped queries.

    Backs `marq collection include/exclude`. Excluding removes it from the
    default scope only: an explicit `-c <name>` still searches it.

    Note:
        Excluding *every* collection leaves the default scope empty, and an
        empty scope matches nothing rather than everything. That direction
        was once inverted - see `_filter_by_collections` in
        cli/commands/read.py.

    Args:
        include: True to put it back in the default scope, False to drop it.

    Raises:
        CollectionNotFoundError: No collection of that name owned by `user`.
        PermissionDeniedError: `can_access()` refused `admin` on it.
    """
    collection = await _resolve_owned_collection(session, user, name, "admin")
    collection.include_by_default = include
    session.add(collection)
    await session.flush()


@dataclass
class CollectionListRow:
    """One row of `list_collections()`, with its document statistics.

    Attributes:
        name: Collection name, unique per owner.
        path: Filesystem path the collection indexes.
        pattern: Glob used when walking that path.
        doc_count: Active documents. Currently identical to
            `active_count` - the distinction is reserved for when
            inactive (deactivated but not yet reclaimed) rows are counted
            separately.
        active_count: Documents currently on disk and indexed.
        last_modified: Newest `modified_at` among the active documents, or
            None for an empty collection.
        include_by_default: Whether unscoped queries search this collection.
    """

    name: str
    path: str
    pattern: str
    doc_count: int
    active_count: int
    last_modified: datetime | None
    include_by_default: bool


async def list_collections(session: AsyncSession, user: CurrentUser) -> list[CollectionListRow]:
    """Every collection `user` can read, with per-collection statistics.

    Today that is all of them, since `can_access()` is mocked True - but
    the filter is real and applied per row, not skipped.

    Returns:
        Rows ordered by collection name.

    Note:
        Runs one statistics query per collection (an N+1). Fine at the
        scale this runs at - a handful of collections per user - but the
        counts would need folding into a single grouped query before that
        stopped being true.
    """
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
