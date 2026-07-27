"""Content/document reclamation - backs the `cleanup` command, and the
orphan sweep that `collection remove` and every reindex run at the end.
"""

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col

from qmd_py.auth import CurrentUser
from qmd_py.db.models import Content, Document, LlmCache
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
    rows."""
    result = await session.execute(
        delete(Content).where(~col(Content.hash).in_(select(col(Document.hash))))
    )
    await session.flush()
    return result.rowcount or 0  # type: ignore[attr-defined]


async def delete_llm_cache(session: AsyncSession) -> int:
    result = await session.execute(delete(LlmCache))
    await session.flush()
    return result.rowcount or 0  # type: ignore[attr-defined]


async def delete_inactive_documents(session: AsyncSession, user: CurrentUser) -> int:
    """Hard-deletes deactivated document rows (not their content - that's
    `cleanup_orphaned_content`'s job) for every collection `user` owns."""
    collection_ids = await resolve_collection_ids(session, user, None)
    if not collection_ids:
        return 0
    result = await session.execute(
        delete(Document).where(
            col(Document.collection_id).in_(collection_ids), ~col(Document.active)
        )
    )
    await session.flush()
    return result.rowcount or 0  # type: ignore[attr-defined]
