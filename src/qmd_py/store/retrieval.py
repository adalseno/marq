"""Document retrieval - get / multi-get / ls / status. Port of the TS
reference's findDocument/multiGet/matchFilesByGlob (src/store.ts,
src/cli/qmd.ts).

Deliberately omits the legacy handelize()/case-insensitive-path migration
lookups and the excluded-by-ignore-rule error variant (ignore_patterns
isn't wired into the indexer).
"""

import fnmatch
import re
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col

from qmd_py.auth import CurrentUser
from qmd_py.db.models import Collection, Content, Document
from qmd_py.search._acl import resolve_collection_ids
from qmd_py.search.fts import get_context_for_path, get_docid
from qmd_py.store._common import _resolve_owned_collection, add_line_numbers
from qmd_py.store.collection import list_collections
from qmd_py.vpath import is_virtual_path, parse_virtual_path

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
    fetched only for the document(s) that actually end up matching).

    Ordered, so every "first match wins" scan over this list is
    reproducible: `find_document` resolves a 6-char docid by hash prefix,
    and two documents can share one. Unordered, which of them you got back
    was down to whatever order Postgres happened to return rows in.
    """
    if not collection_ids:
        return []
    result = await session.execute(
        select(Document, col(Collection.name))
        .join(Collection, col(Collection.id) == Document.collection_id)
        .where(col(Document.active), col(Document.collection_id).in_(collection_ids))
        .order_by(col(Collection.name), col(Document.path))
    )
    return [(doc, name) for doc, name in result.all()]


def _match_named_document(
    refs: list[tuple[Document, str]], name: str
) -> tuple[Document, str] | None:
    """Exact virtual path (marq://collection/path), then exact bare path,
    then a suffix match against the full virtual path (`marq://coll/path`,
    not just the bare path - so e.g. "sample/src/foo.py" or a partial
    "src/foo.py" both resolve) - in that order, first match wins, same as
    the TS reference's `findDocument`.

    The suffix match requires a `/` segment boundary - a deliberate
    behavior improvement over the TS reference, whose bare `endsWith`
    lets `o.py` silently resolve to `src/foo.py` mid-filename. Every
    documented partial-path form still works; a genuine miss falls
    through to the caller's "did you mean" suggestions."""
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
        if f"marq://{coll_name}/{document.path}".endswith(f"/{name}"):
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
    """One resolved document, body included.

    Carries the path in three forms because different surfaces want
    different ones - MCP resources use the URI, CLI output the readable
    form, glob matching all three.

    Attributes:
        filepath: Virtual URI, `marq://<collection>/<path>`.
        display_path: `<collection>/<path>` - what CLI output prints.
        title: Extracted heading, or the filename stem as fallback.
        context: Hierarchical context for this path (global first, then
            each matching prefix, most general first), or None if none
            applies.
        hash: Full SHA-256 of the body.
        docid: First 6 chars of `hash` - what `#abc123` lookups use. Short
            enough to collide in principle; resolution is deterministic.
        collection_name: Owning collection.
        modified_at: Source file's mtime at index time, not the row's.
        body_length: `len(body)` in characters, not bytes.
        body: Full document text.
    """

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
    """Returned instead of raising when a lookup finds nothing.

    A miss is an ordinary outcome here - the caller is usually a person
    mistyping a path - so it is a value to render, not an exception.

    Attributes:
        query: The lookup string as given, so the caller can echo it back.
        similar_files: Up to five paths within Levenshtein distance 5, for
            a "did you mean" hint. Empty when the query looked like a
            docid, since edit distance over hex is meaningless.
    """

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
    """Resolve one document by docid, virtual path, or bare path.

    Tries, in order: docid (`#abc123` or bare hex, matched as a hash
    prefix), exact virtual path (`marq://collection/path`), exact bare
    path, then a suffix match against the full virtual path - so both
    `sample/src/foo.py` and a partial `src/foo.py` resolve. Port of the TS
    reference's `findDocument`.

    Args:
        filename: Docid or path, in any of the forms above.
        collection_name: Restrict to one collection. None searches every
            collection the user can read.

    Returns:
        A `DocumentDetail` with the body, or a `DocumentNotFound` carrying
        near-miss suggestions. Never raises for a miss.

    Note:
        A 6-char docid can front two documents. Which one wins is
        arbitrary but stable: `_active_document_refs` is ordered, so
        repeated lookups agree rather than depending on row order.
    """
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
    """One glob hit, without its body.

    Attributes:
        filepath: Virtual URI, `marq://<collection>/<path>`.
        display_path: Bare relative path, *not* collection-prefixed -
            ambiguous across collections, but kept for parity with the TS
            reference's own convention.
        body_length: Character count, fetched with `length()` in SQL so
            the body itself is never loaded.
    """

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
    """One document from a `multi_get()` batch, possibly skipped.

    Attributes:
        filepath: Virtual URI, `marq://<collection>/<path>`.
        display_path: `<collection>/<path>`.
        title: Extracted heading or filename stem.
        body: Document text, already truncated and line-numbered per the
            call's arguments. Empty string when `skipped`.
        context: Hierarchical context for this path, or None.
        skipped: True when the file exceeded `max_bytes` and was not read.
        docid: Six-char hash prefix, for a follow-up `get`.
        skip_reason: Human-readable explanation, set only when `skipped`.
    """

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
    port of the TS reference's `multiGet` (src/cli/qmd.ts).

    The two forms are mutually exclusive, not combinable: `pattern` counts
    as a comma-separated list only if it holds no glob metacharacter at
    all. So `"a.md,b*.md"` is treated as one glob containing a literal
    comma and matches nothing, rather than as two patterns. That's the TS
    reference's behavior, kept deliberately for parity - split such a
    request into separate calls.

    Oversized files are reported rather than dropped: the entry comes back
    with `skipped=True` and a `skip_reason`, so a caller can tell "too
    large, fetch it singly" from "no such file".

    Args:
        pattern: Glob, or a comma-separated list of paths/docids.
        max_lines: Truncate each body to this many lines, appending a
            note about how many were omitted. None keeps the whole body.
        max_bytes: Skip any document longer than this rather than
            returning it. Compared against the character length.
        line_numbers: Prefix each body line with `N: `.

    Returns:
        One entry per match, in the order the documents were scanned.
        Empty when nothing matched - a miss is not an error here.
    """
    collection_ids = await resolve_collection_ids(session, user, None)
    if not collection_ids:
        return []

    refs = await _active_document_refs(session, collection_ids)
    is_comma_separated = "," in pattern and not any(c in pattern for c in "*?{[")

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
    """One row of `list_files()` - metadata only, no body.

    Attributes:
        path: Path relative to the collection root.
        title: Extracted heading or filename stem.
        modified_at: Source file's mtime as recorded at index time.
        size: Body length in characters, computed in SQL.
    """

    path: str
    title: str
    modified_at: datetime
    size: int


async def list_files(
    session: AsyncSession, user: CurrentUser, collection_name: str, path_prefix: str | None = None
) -> list[FileRow]:
    """List one collection's active files, optionally under a sub-path.

    Backs `marq ls <collection>[/path]`.

    Args:
        path_prefix: Restrict to paths starting with this string. Matched
            as a literal prefix (LIKE wildcards `%`/`_` are escaped), not
            as a glob, so `*` has no special meaning either.

    Returns:
        Rows ordered by path. Empty for a collection with no matches -
        only an unknown collection raises.

    Raises:
        CollectionNotFoundError: No collection of that name owned by `user`.
        PermissionDeniedError: `can_access()` refused `read` on it.
    """
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
        stmt = stmt.where(col(Document.path).startswith(path_prefix, autoescape=True))
    stmt = stmt.order_by(col(Document.path))
    rows = await session.execute(stmt)
    return [FileRow(path=p, title=t, modified_at=m, size=s) for p, t, m, s in rows]


@dataclass
class CollectionStatus:
    """Per-collection line of `get_status()`.

    Attributes:
        name: Collection name.
        path: Indexed filesystem path.
        pattern: Glob used to walk it.
        doc_count: Active documents.
        last_updated: Newest document mtime, or None when empty.
    """

    name: str
    path: str
    pattern: str
    doc_count: int
    last_updated: datetime | None


@dataclass
class StatusInfo:
    """Index summary behind `marq status`.

    Attributes:
        total_documents: Active documents across every readable
            collection.
        collections: Per-collection detail, ordered by name.
    """

    total_documents: int
    collections: list[CollectionStatus]


async def get_status(session: AsyncSession, user: CurrentUser) -> StatusInfo:
    """Backs `status` - reuses `list_collections`'s per-collection stats
    rather than recomputing them. Deliberately narrower than the TS
    reference's own `status` command for now (no MCP-daemon liveness, no
    AST-chunking availability, no embedding-completeness/vector-index-
    health checks) - those need Phase 9's MCP server and Phase 5's
    embedding pipeline wired into the CLI first.

    Returns:
        Totals and per-collection detail; see `StatusInfo`. Counts only
        collections the user can read, so it is empty rather than an error
        for a user with none.
    """
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
