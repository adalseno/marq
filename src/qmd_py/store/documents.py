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
    """Store a body under its content hash, if not already present.

    Idempotent by design: content is addressed by hash and shared across
    every collection, so re-indexing an unchanged file - or the same file
    in a second worktree - inserts nothing and costs nothing.

    Args:
        hash_: SHA-256 hex digest of `doc`, from `hash_content()`. Not
            recomputed or verified here.
        doc: The full body text.
    """
    stmt = (
        pg_insert(Content)
        .values(hash=hash_, doc=doc)
        .on_conflict_do_nothing(index_elements=["hash"])
    )
    await session.execute(stmt)


async def find_active_document(
    session: AsyncSession, collection_id: int, path: str
) -> Document | None:
    """Find the live document at a path, ignoring deactivated rows.

    Returns:
        The document, or None if there is none or it has been
        deactivated. Use `find_document_by_path()` when a deactivated row
        still matters.
    """
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
    """Find the document at a path whether or not it is active.

    `reindex_collection` needs this rather than `find_active_document`, to
    reactivate a deactivated document whose file reappears instead of
    violating `Document`'s table-wide `UNIQUE(collection_id, path)`
    constraint by inserting a second row.

    Returns:
        The document, active or not, or None if the path was never
        indexed.
    """
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
    """Create a document row and build its search vector.

    Assumes the content row already exists - call `insert_content()`
    first, since `document.hash` is a NOT NULL foreign key.

    Args:
        path: Path relative to the collection root.
        hash_: Content hash linking to the stored body.
        created_at: Usually the file's mtime rather than now, so the value
            survives a reindex from a fresh clone.
        modified_at: The file's mtime.

    Returns:
        The persisted row, with `id` populated by the flush.

    Raises:
        sqlalchemy.exc.IntegrityError: A row already exists for this
            `(collection_id, path)`, active or not, or `hash_` has no
            content row.
    """
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
    """Point a document at new content and rebuild its search vector.

    Reactivates it (`active=True`) unconditionally: every caller reaches
    this because a real file on disk currently maps to this
    `(collection_id, path)`, including a file that reappeared after being
    deactivated - switching git branches back and forth over the same
    indexed working tree does exactly that. `Document` has a table-wide
    `UNIQUE(collection_id, path)` with no partial/active condition, so
    there can only ever be one row per path regardless of active status;
    reactivating it is the only option, inserting a second is not.

    Args:
        document: The row to update, already loaded in this session.
        hash_: New content hash. Its content row must exist.
        modified_at: The file's current mtime.
    """
    document.title = title
    document.hash = hash_
    document.modified_at = modified_at
    document.active = True
    session.add(document)
    await session.flush()
    await update_document_search_vector(session, document.id)


async def deactivate_document(session: AsyncSession, collection_id: int, path: str) -> None:
    """Mark a document inactive, keeping the row.

    Soft delete: search and retrieval ignore inactive documents, but the
    row survives so a returning file can reactivate it rather than
    colliding with `UNIQUE(collection_id, path)`. `marq cleanup` is what
    hard-deletes them.

    A path with no row is a no-op, not an error.
    """
    await session.execute(
        update(Document)
        .where(col(Document.collection_id) == collection_id, col(Document.path) == path)
        .values(active=False)
    )
    await session.flush()


async def get_active_document_paths(session: AsyncSession, collection_id: int) -> list[str]:
    """List the paths currently indexed and active in a collection.

    Returns:
        Paths relative to the collection root, unordered. `reindex_collection`
        diffs this against what it found on disk to decide what to
        deactivate.
    """
    result = await session.execute(
        select(col(Document.path)).where(
            col(Document.collection_id) == collection_id, col(Document.active)
        )
    )
    return [p for (p,) in result.all()]
