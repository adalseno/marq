"""Service/CRUD layer - the single facade the CLI, MCP server, and any
future REST API all call through (fixes the TS reference implementation's
inconsistency #3, where the SDK's `searchLex`/`searchVector` silently
skipped behavior the CLI's `search`/`vsearch` commands did: one code path
per operation here, not two that can drift apart).

Every function that touches a specific collection's data resolves the
`Collection` row and calls `can_access()` (see auth.py) before proceeding -
mocked to always allow today, but the choke point is real from day one.
"""

import fnmatch
import glob as glob_module
import hashlib
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import delete, func, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col

from qmd_py.auth import CurrentUser, can_access
from qmd_py.db.models import (
    Collection,
    CollectionContext,
    CollectionGrant,
    Content,
    Document,
    LlmCache,
    User,
)
from qmd_py.search._acl import resolve_collection_ids
from qmd_py.search.fts import get_context_for_path, get_docid, update_document_search_vector
from qmd_py.vpath import is_virtual_path, parse_virtual_path


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


def utcnow() -> datetime:
    return datetime.now(UTC)


# =============================================================================
# Indexing (`update` / `collection add`) - port of the TS reference's
# reindexCollection (src/store.ts). Deliberately omits the legacy
# handelize()/case-insensitive-path migration lookup (findOrMigrateLegacyDocument)
# - see the module docstring's earlier note; that only existed to migrate very
# old SQLite indexes, which doesn't apply to a from-scratch project.
# =============================================================================

_EXCLUDE_DIRS = frozenset({"node_modules", ".git", ".cache", "vendor", "dist", "build"})


def _expand_braces(pattern: str) -> list[str]:
    """Expands one `{a,b,c}` alternation group - Python's glob has no
    brace-expansion support (that's a shell feature), but our collection
    patterns (e.g. `**/*.{py,md,json}`) need it."""
    match = re.search(r"\{([^{}]+)\}", pattern)
    if not match:
        return [pattern]
    prefix, suffix = pattern[: match.start()], pattern[match.end() :]
    return [prefix + alt + suffix for alt in match.group(1).split(",")]


def _discover_files(
    collection_path: str, pattern: str, ignore_patterns: list[str] | None
) -> list[str]:
    """Relative paths (glob-matched, hidden-file/dir and _EXCLUDE_DIRS
    filtered) under `collection_path` - port of reindexCollection's
    fast-glob call + hidden-file filter."""
    root = Path(collection_path)
    seen: set[str] = set()
    files: list[str] = []
    for expanded in _expand_braces(pattern):
        for match in glob_module.glob(
            expanded, root_dir=str(root), recursive=True, include_hidden=False
        ):
            rel_path = match
            if rel_path in seen:
                continue
            parts = rel_path.split("/")
            if any(p in _EXCLUDE_DIRS or p.startswith(".") for p in parts):
                continue
            if ignore_patterns and any(fnmatch.fnmatch(rel_path, ip) for ip in ignore_patterns):
                continue
            if not (root / rel_path).is_file():
                continue
            seen.add(rel_path)
            files.append(rel_path)
    return sorted(files)


@dataclass
class ReindexResult:
    indexed: int
    updated: int
    unchanged: int
    removed: int
    orphaned_cleaned: int


async def reindex_collection(
    session: AsyncSession, user: CurrentUser, name: str
) -> ReindexResult:
    """Walk a collection's filesystem path, syncing `document`/`content`
    to match what's on disk - backs `collection add` and `update`.

    Simplification vs. the TS reference: a title-only change (same
    content hash) is counted as "updated" here rather than its own
    separate `updated`-for-title-only bucket - these are just informational
    summary counts shown to the user, not load-bearing for correctness.
    """
    collection = await _resolve_owned_collection(session, user, name, "write")
    files = _discover_files(collection.path, collection.pattern, collection.ignore_patterns)
    root = Path(collection.path)
    indexed = 0
    updated = 0
    unchanged = 0
    seen_paths: set[str] = set()

    for rel_path in files:
        filepath = root / rel_path
        try:
            content = filepath.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if not content.strip():
            continue
        seen_paths.add(rel_path)

        digest = await hash_content(content)
        title = extract_title(content, rel_path)
        modified_at = datetime.fromtimestamp(filepath.stat().st_mtime, tz=UTC)

        existing = await find_document_by_path(session, collection.id, rel_path)
        if existing is not None:
            if existing.active and existing.hash == digest and existing.title == title:
                unchanged += 1
                continue
            if existing.hash != digest:
                await insert_content(session, digest, content)
            await update_document(session, existing, title, digest, modified_at)
            updated += 1
        else:
            await insert_content(session, digest, content)
            await insert_document(
                session, collection.id, rel_path, title, digest, modified_at, modified_at
            )
            indexed += 1

    removed = 0
    for active_path in await get_active_document_paths(session, collection.id):
        if active_path not in seen_paths:
            await deactivate_document(session, collection.id, active_path)
            removed += 1

    orphaned = await cleanup_orphaned_content(session)

    return ReindexResult(
        indexed=indexed,
        updated=updated,
        unchanged=unchanged,
        removed=removed,
        orphaned_cleaned=orphaned,
    )


# =============================================================================
# Cleanup (`cleanup` command)
# =============================================================================


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


def add_line_numbers(text: str, start_line: int = 1) -> str:
    lines = text.split("\n")
    return "\n".join(f"{start_line + i}: {line}" for i, line in enumerate(lines))


# =============================================================================
# Document retrieval (get / multi-get / ls) - port of the TS reference's
# findDocument/multiGet/matchFilesByGlob (src/store.ts, src/cli/qmd.ts).
#
# Deliberately omits: the legacy handelize()/case-insensitive-path migration
# lookups (see the module docstring's "not implemented" note in the plan -
# these only existed to migrate very old SQLite indexes) and the
# excluded-by-ignore-rule error variant (ignore_patterns isn't wired into
# an actual indexer yet - that's Phase 7's job).
# =============================================================================

_DOCID_PATTERN = re.compile(r"^[0-9a-f]{6,}$", re.IGNORECASE)


def _looks_like_docid(candidate: str) -> bool:
    return bool(_DOCID_PATTERN.match(candidate))


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    previous_row = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current_row = [i] + [0] * len(b)
        for j, char_b in enumerate(b, start=1):
            cost = 0 if char_a == char_b else 1
            current_row[j] = min(
                previous_row[j] + 1, current_row[j - 1] + 1, previous_row[j - 1] + cost
            )
        previous_row = current_row
    return previous_row[-1]


def _find_similar_paths(
    filename: str, paths: list[str], max_distance: int = 5, limit: int = 5
) -> list[str]:
    scored = [(path, _levenshtein(filename, path)) for path in paths]
    close_enough = sorted((s for s in scored if s[1] <= max_distance), key=lambda s: s[1])
    return [path for path, _ in close_enough[:limit]]


async def _active_document_refs(
    session: AsyncSession, collection_ids: list[int]
) -> list[tuple[Document, str]]:
    """(Document, owning collection name) for every active document in
    `collection_ids` - deliberately excludes body (lives in `Content`,
    fetched only for the document(s) that actually end up matching)."""
    if not collection_ids:
        return []
    result = await session.execute(
        select(Document, col(Collection.name))
        .join(Collection, col(Collection.id) == Document.collection_id)
        .where(col(Document.active), col(Document.collection_id).in_(collection_ids))
    )
    return [(doc, name) for doc, name in result.all()]


def _match_named_document(
    refs: list[tuple[Document, str]], name: str
) -> tuple[Document, str] | None:
    """Exact virtual path (marq://collection/path), then exact bare path,
    then a suffix match against the full virtual path (`marq://coll/path`,
    not just the bare path - so e.g. "sample/src/foo.py" or a partial
    "src/foo.py" both resolve) - in that order, first match wins, same as
    the TS reference's `findDocument`."""
    parsed = parse_virtual_path(name) if is_virtual_path(name) else None
    if parsed is not None:
        target_collection, path = parsed
        for document, coll_name in refs:
            if coll_name == target_collection and document.path == path:
                return document, coll_name
        return None

    for document, coll_name in refs:
        if document.path == name:
            return document, coll_name
    for document, coll_name in refs:
        if f"marq://{coll_name}/{document.path}".endswith(name):
            return document, coll_name
    return None


async def _document_body(session: AsyncSession, hash_: str) -> str:
    return (
        await session.execute(select(col(Content.doc)).where(col(Content.hash) == hash_))
    ).scalar_one()


async def _document_body_length(session: AsyncSession, hash_: str) -> int:
    length = (
        await session.execute(
            select(func.length(col(Content.doc))).where(col(Content.hash) == hash_)
        )
    ).scalar_one()
    return int(length)


@dataclass
class DocumentDetail:
    filepath: str
    display_path: str
    title: str
    context: str | None
    hash: str
    docid: str
    collection_name: str
    modified_at: datetime
    body_length: int
    body: str


@dataclass
class DocumentNotFound:
    query: str
    similar_files: list[str]


async def _build_document_detail(
    session: AsyncSession, user: CurrentUser, document: Document, collection_name: str
) -> DocumentDetail:
    body = await _document_body(session, document.hash)
    context = await get_context_for_path(session, user, document.collection_id, document.path)
    return DocumentDetail(
        filepath=f"marq://{collection_name}/{document.path}",
        display_path=f"{collection_name}/{document.path}",
        title=document.title,
        context=context,
        hash=document.hash,
        docid=get_docid(document.hash),
        collection_name=collection_name,
        modified_at=document.modified_at,
        body_length=len(body),
        body=body,
    )


async def find_document(
    session: AsyncSession, user: CurrentUser, filename: str, collection_name: str | None = None
) -> DocumentDetail | DocumentNotFound:
    """Resolve a document by docid (`#abc123` or bare hex), virtual path
    (`marq://collection/path`), or bare path (exact match, then suffix
    match) - port of the TS reference's `findDocument`."""
    collection_ids = await resolve_collection_ids(session, user, collection_name)
    if not collection_ids:
        return DocumentNotFound(query=filename, similar_files=[])

    refs = await _active_document_refs(session, collection_ids)

    stripped = filename[1:] if filename.startswith("#") else filename
    if _looks_like_docid(stripped):
        lowered = stripped.lower()
        for document, coll_name in refs:
            if document.hash.lower().startswith(lowered):
                return await _build_document_detail(session, user, document, coll_name)
        return DocumentNotFound(query=filename, similar_files=[])

    found = _match_named_document(refs, filename)
    if found is not None:
        document, coll_name = found
        return await _build_document_detail(session, user, document, coll_name)

    similar = _find_similar_paths(filename, [d.path for d, _ in refs])
    return DocumentNotFound(query=filename, similar_files=similar)


@dataclass
class GlobMatch:
    filepath: str
    display_path: str
    body_length: int


def _matches_pattern(document: Document, collection_name: str, pattern: str) -> bool:
    virtual_path = f"marq://{collection_name}/{document.path}"
    collection_path = f"{collection_name}/{document.path}"
    return (
        fnmatch.fnmatchcase(virtual_path, pattern)
        or fnmatch.fnmatchcase(document.path, pattern)
        or fnmatch.fnmatchcase(collection_path, pattern)
    )


async def match_files_by_glob(
    session: AsyncSession, user: CurrentUser, pattern: str
) -> list[GlobMatch]:
    """Glob-match against three forms of every active, accessible
    document's path (virtual `marq://collection/path`, bare path, and
    `collection/path`) - matches if any form matches, same as the TS
    reference's `matchFilesByGlob`. `display_path` is the bare relative
    path (not collection-prefixed), matching that function's own
    (slightly ambiguous across collections) convention."""
    collection_ids = await resolve_collection_ids(session, user, None)
    refs = await _active_document_refs(session, collection_ids)
    matches = []
    for document, coll_name in refs:
        if _matches_pattern(document, coll_name, pattern):
            body_length = await _document_body_length(session, document.hash)
            matches.append(
                GlobMatch(
                    filepath=f"marq://{coll_name}/{document.path}",
                    display_path=document.path,
                    body_length=body_length,
                )
            )
    return matches


@dataclass
class MultiGetFile:
    filepath: str
    display_path: str
    title: str
    body: str
    context: str | None
    skipped: bool
    docid: str | None = None
    skip_reason: str | None = None


DEFAULT_MULTI_GET_MAX_BYTES = 65536
"""The TS reference's real default - its own --help text says "10KB", but
the code's actual default is 64KB (one of the plan's flagged
inconsistencies to fix, not reproduce)."""


async def multi_get(
    session: AsyncSession,
    user: CurrentUser,
    pattern: str,
    max_lines: int | None = None,
    max_bytes: int = DEFAULT_MULTI_GET_MAX_BYTES,
    line_numbers: bool = True,
) -> list[MultiGetFile]:
    """Fetch multiple documents by glob pattern or comma-separated list -
    port of the TS reference's `multiGet` (src/cli/qmd.ts)."""
    collection_ids = await resolve_collection_ids(session, user, None)
    if not collection_ids:
        return []

    refs = await _active_document_refs(session, collection_ids)
    is_comma_separated = "," in pattern and not any(c in pattern for c in "*?{")

    matched: list[tuple[Document, str]] = []
    if is_comma_separated:
        for name in (n.strip() for n in pattern.split(",") if n.strip()):
            found = _match_named_document(refs, name)
            if found is not None:
                matched.append(found)
    else:
        matched = [
            (document, coll_name)
            for document, coll_name in refs
            if _matches_pattern(document, coll_name, pattern)
        ]

    results: list[MultiGetFile] = []
    for document, coll_name in matched:
        filepath = f"marq://{coll_name}/{document.path}"
        display_path = f"{coll_name}/{document.path}"
        docid = get_docid(document.hash)
        context = await get_context_for_path(session, user, document.collection_id, document.path)

        body_length = await _document_body_length(session, document.hash)
        if body_length > max_bytes:
            results.append(
                MultiGetFile(
                    filepath=filepath,
                    display_path=display_path,
                    title=document.title,
                    body="",
                    context=context,
                    skipped=True,
                    docid=docid,
                    skip_reason=(
                        f"File too large ({body_length // 1024}KB > {max_bytes // 1024}KB). "
                        f"Use 'marq get {display_path}' to retrieve."
                    ),
                )
            )
            continue

        body = await _document_body(session, document.hash)
        if max_lines is not None:
            lines = body.split("\n")
            if len(lines) > max_lines:
                omitted = len(lines) - max_lines
                body = "\n".join(lines[:max_lines]) + f"\n\n[... truncated {omitted} more lines]"
        if line_numbers:
            body = add_line_numbers(body)

        results.append(
            MultiGetFile(
                filepath=filepath,
                display_path=display_path,
                title=document.title,
                body=body,
                context=context,
                skipped=False,
                docid=docid,
            )
        )

    return results


@dataclass
class FileRow:
    path: str
    title: str
    modified_at: datetime
    size: int


async def list_files(
    session: AsyncSession, user: CurrentUser, collection_name: str, path_prefix: str | None = None
) -> list[FileRow]:
    """Files in one collection, optionally under a sub-path - backs `ls
    <collection>[/path]`."""
    collection = await _resolve_owned_collection(session, user, collection_name, "read")
    stmt = (
        select(
            col(Document.path),
            col(Document.title),
            col(Document.modified_at),
            func.length(col(Content.doc)),
        )
        .join(Content, col(Content.hash) == col(Document.hash))
        .where(col(Document.collection_id) == collection.id, col(Document.active))
    )
    if path_prefix:
        stmt = stmt.where(col(Document.path).like(f"{path_prefix}%"))
    stmt = stmt.order_by(col(Document.path))
    rows = await session.execute(stmt)
    return [FileRow(path=p, title=t, modified_at=m, size=s) for p, t, m, s in rows]


@dataclass
class CollectionStatus:
    name: str
    path: str
    pattern: str
    doc_count: int
    last_updated: datetime | None


@dataclass
class StatusInfo:
    total_documents: int
    collections: list[CollectionStatus]


async def get_status(session: AsyncSession, user: CurrentUser) -> StatusInfo:
    """Backs `status` - reuses `list_collections`'s per-collection stats
    rather than recomputing them. Deliberately narrower than the TS
    reference's own `status` command for now (no MCP-daemon liveness, no
    AST-chunking availability, no embedding-completeness/vector-index-
    health checks) - those need Phase 9's MCP server and Phase 5's
    embedding pipeline wired into the CLI first."""
    collections = await list_collections(session, user)
    return StatusInfo(
        total_documents=sum(c.active_count for c in collections),
        collections=[
            CollectionStatus(
                name=c.name,
                path=c.path,
                pattern=c.pattern,
                doc_count=c.active_count,
                last_updated=c.last_modified,
            )
            for c in collections
        ],
    )
