"""Content/document reclamation - backs the `cleanup` command, and the
orphan sweep that `collection remove` and every reindex run at the end.
"""

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col

from qmd_py.auth import CurrentUser
from qmd_py.db.models import Content, Document, LlmCache
from qmd_py.db.result import affected_rows
from qmd_py.search._acl import resolve_collection_ids


async def cleanup_orphaned_content(session: AsyncSession) -> int:
    """A hash is only truly orphaned once NO document row - active or
    inactive - references it: `document.hash` is a NOT NULL FK with no
    `ON DELETE CASCADE`, so Postgres blocks deleting a content row an
    *inactive* document still points to, even though search/retrieval
    never reads inactive documents. Filtering this query to only active
    documents (as an earlier version did) raised a FK violation the first
    time an inactive document was the last remaining reference. Run
    `delete_inactive_documents` first (see the `cleanup` command) to
    actually reclaim content orphaned only by now-abandoned inactive
    rows.

    Returns:
        Number of content rows deleted.
    """
    result = await session.execute(
        delete(Content).where(~col(Content.hash).in_(select(col(Document.hash))))
    )
    await session.flush()
    return affected_rows(result)


async def delete_llm_cache(session: AsyncSession) -> int:
    """Empty the cached-LLM-response table.

    Global, not per user: the cache is keyed by prompt, so there is
    nothing user-scoped to preserve.

    Returns:
        Number of cached responses deleted.
    """
    result = await session.execute(delete(LlmCache))
    await session.flush()
    return affected_rows(result)


async def delete_inactive_documents(session: AsyncSession, user: CurrentUser) -> int:
    """Hard-delete deactivated document rows across a user's collections.

    Removes the rows only; their content is `cleanup_orphaned_content`'s
    job, and has to run *after* this to reclaim bodies whose last
    remaining reference was one of these inactive rows.

    Deliberately irreversible: once these are gone, a file that reappears
    is indexed as a new document rather than reactivating its old row.

    Returns:
        Number of document rows deleted; 0 when the user has no
        accessible collections.
    """
    collection_ids = await resolve_collection_ids(session, user, None)
    if not collection_ids:
        return 0
    result = await session.execute(
        delete(Document).where(
            col(Document.collection_id).in_(collection_ids), ~col(Document.active)
        )
    )
    await session.flush()
    return affected_rows(result)
