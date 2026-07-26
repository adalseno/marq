"""Service/CRUD layer - the single facade the CLI, MCP server, and any
future REST API all call through (fixes the TS reference implementation's
inconsistency #3, where the SDK's `searchLex`/`searchVector` silently
skipped behavior the CLI's `search`/`vsearch` commands did: one code path
per operation here, not two that can drift apart).

Every function that touches a specific collection's data resolves the
`Collection` row and calls `can_access()` (see auth.py) before proceeding -
mocked to always allow today, but the choke point is real from day one.
"""

import hashlib
import re
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import delete, func, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col

from qmd_py.auth import CurrentUser, can_access
from qmd_py.db.models import Collection, CollectionContext, Content, Document, User


class CollectionNotFoundError(Exception):
    pass


class PermissionDeniedError(Exception):
    pass


async def hash_content(content: str) -> str:
    """SHA256 hex digest - matches the TS reference's `hashContent()`
    exactly (content-addressable hashes must agree byte-for-byte with the
    TS side for the Phase 3 parity check to mean anything)."""
    return hashlib.sha256(content.encode()).hexdigest()


_MD_HEADING = re.compile(r"^##?\s+(.+)$", re.MULTILINE)
_ORG_TITLE_PROP = re.compile(r"^#\+TITLE:\s*(.+)$", re.MULTILINE | re.IGNORECASE)
_ORG_HEADING = re.compile(r"^\*+\s+(.+)$", re.MULTILINE)


def extract_title(content: str, filename: str) -> str:
    """Port of the TS reference's `extractTitle()`: first `#`/`##` markdown
    heading (skipping a generic "Notes" heading in favor of the next one,
    a quirk carried over from the TS version's own note-taking-app
    interop), `#+TITLE:`/org heading for `.org` files, else the filename
    without its extension."""
    ext = filename[filename.rfind(".") :].lower() if "." in filename else ""

    if ext == ".md":
        match = _MD_HEADING.search(content)
        if match:
            title = match.group(1).strip()
            if title in ("\U0001f4dd Notes", "Notes"):
                next_match = re.search(r"^##\s+(.+)$", content, re.MULTILINE)
                if next_match:
                    return next_match.group(1).strip()
            return title
    elif ext == ".org":
        prop_match = _ORG_TITLE_PROP.search(content)
        if prop_match:
            return prop_match.group(1).strip()
        heading_match = _ORG_HEADING.search(content)
        if heading_match:
            return heading_match.group(1).strip()

    stem = re.sub(r"\.[^.]+$", "", filename)
    return stem.rsplit("/", 1)[-1] or filename


# =============================================================================
# Collections
# =============================================================================


async def _resolve_owned_collection(
    session: AsyncSession, user: CurrentUser, name: str, permission: str = "read"
) -> Collection:
    result = await session.execute(
        select(Collection).where(
            col(Collection.owner_user_id) == user.id, col(Collection.name) == name
        )
    )
    collection = result.scalar_one_or_none()
    if collection is None:
        raise CollectionNotFoundError(name)
    if not await can_access(user, collection, permission):
        raise PermissionDeniedError(name)
    return collection


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
    collection = await _resolve_owned_collection(session, user, name, "admin")
    deleted = await session.execute(
        delete(Document).where(col(Document.collection_id) == collection.id)
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
        if not await can_access(user, collection, "read"):
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


# =============================================================================
# Context
# =============================================================================


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


# =============================================================================
# Content-addressable ingest
# =============================================================================


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
    return document


async def update_document(
    session: AsyncSession, document: Document, title: str, hash_: str, modified_at: datetime
) -> None:
    document.title = title
    document.hash = hash_
    document.modified_at = modified_at
    session.add(document)
    await session.flush()


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


async def cleanup_orphaned_content(session: AsyncSession) -> int:
    result = await session.execute(
        delete(Content).where(
            ~col(Content.hash).in_(select(col(Document.hash)).where(col(Document.active)))
        )
    )
    await session.flush()
    return result.rowcount or 0  # type: ignore[attr-defined]


def utcnow() -> datetime:
    return datetime.now(UTC)
