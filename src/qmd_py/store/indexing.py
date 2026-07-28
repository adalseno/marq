"""Filesystem walk and reindex - backs `collection add` and `update`.
Port of the TS reference's reindexCollection (src/store.ts).

Deliberately omits the legacy handelize()/case-insensitive-path migration
lookup (findOrMigrateLegacyDocument): that only existed to migrate very
old SQLite indexes, which doesn't apply to a from-scratch project.
"""

import fnmatch
import glob as glob_module
import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from qmd_py.auth import CurrentUser
from qmd_py.log import log_duration
from qmd_py.store._common import _resolve_owned_collection, extract_title, hash_content
from qmd_py.store.cleanup import cleanup_orphaned_content
from qmd_py.store.documents import (
    deactivate_document,
    find_document_by_path,
    get_active_document_paths,
    insert_content,
    insert_document,
    update_document,
)

logger = logging.getLogger(__name__)

_EXCLUDE_DIRS = frozenset({"node_modules", ".git", ".cache", "vendor", "dist", "build"})

MAX_INDEXABLE_CHARS = 1_000_000
"""Documents longer than this are skipped at index time (soft-skip, counted
in `ReindexResult.skipped_oversize`), the same convention `multi_get`'s
`max_bytes` uses. The line sits where Postgres draws it: a tsvector caps at
1 MiB, and a distinct-token-heavy file (log dump, minified code) crosses
that well under 2 MB of source text - feeding it through would abort the
whole reindex with a raw `ProgrammingError`. Even a document that squeaked
under the tsvector cap would only degrade search quality anyway."""


# Matches the leftmost *innermost* alternation group (the character class
# excludes braces, so an outer `{a,{b,c}}` can't match until its inner group
# has been expanded away).
_BRACE_GROUP = re.compile(r"\{([^{}]+)\}")

_MAX_BRACE_EXPANSIONS = 1000
"""Ceiling on the expansion of one pattern. Groups multiply
(`{a,b}/{c,d}/{e,f}` is 8 patterns, each its own filesystem walk), so a
pathological pattern could otherwise turn `update` into a hang."""


def _expand_braces(pattern: str) -> list[str]:
    """Expands every `{a,b,c}` alternation group into the full cross
    product - Python's glob has no brace-expansion support (that's a shell
    feature), but our collection patterns (e.g. `{src,docs}/**/*.{py,md}`)
    need it.

    Expanding only the first group (as an earlier version did) left any
    further group in the returned patterns as a literal `{...}`, which glob
    matches against nothing - a collection with a two-group pattern
    silently indexed zero files.

    An unbalanced `{` has no group to match and is passed through
    untouched, reaching glob as the literal character it already was.
    """
    match = _BRACE_GROUP.search(pattern)
    if match is None:
        return [pattern]

    prefix, suffix = pattern[: match.start()], pattern[match.end() :]
    expansions: list[str] = []
    seen: set[str] = set()
    for alternative in match.group(1).split(","):
        # Recurse: `prefix + alternative + suffix` can still hold further
        # groups, either alongside this one or nested inside it.
        for expanded in _expand_braces(prefix + alternative + suffix):
            if expanded in seen:
                continue
            seen.add(expanded)
            expansions.append(expanded)
            if len(expansions) >= _MAX_BRACE_EXPANSIONS:
                return expansions
    return expansions


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
    """Summary counts from one `reindex_collection()` pass.

    Informational only - the CLI prints them - so they are not relied on
    for correctness anywhere. Files skipped as unreadable or blank appear
    in no bucket at all, so the four document counts need not sum to the
    number of files on disk.

    Attributes:
        indexed: Files that had no document row and got one.
        updated: Existing documents whose content or title changed. Also
            counts a file reappearing after deactivation, and a title-only
            change with an unchanged body hash.
        unchanged: Active documents whose hash and title both matched, so
            nothing was written.
        removed: Documents deactivated because their file is no longer on
            disk. Deactivated, not deleted - `marq cleanup` reclaims them.
        orphaned_cleaned: Content rows dropped because no document, active
            or inactive, still referenced them.
        skipped_oversize: Files skipped for exceeding `MAX_INDEXABLE_CHARS`
            - the one skip reason that gets its own visible count, because
            silently dropping a legitimate (if huge) file from the index
            would otherwise look like a search bug.
    """

    indexed: int
    updated: int
    unchanged: int
    removed: int
    orphaned_cleaned: int
    skipped_oversize: int = 0


async def reindex_collection(
    session: AsyncSession, user: CurrentUser, name: str
) -> ReindexResult:
    """Walk a collection's filesystem path, syncing `document`/`content`
    to match what's on disk - backs `collection add` and `update`.

    Disk is the source of truth for one direction only: a file that has
    vanished deactivates its document rather than deleting it, so the row
    survives to be reactivated if the file comes back (switching git
    branches over an indexed working tree does exactly that). `Document`
    has a table-wide `UNIQUE(collection_id, path)` with no active/inactive
    carve-out, so reactivating the existing row is the only option -
    inserting a second one is impossible.

    Simplification vs. the TS reference: a title-only change (same content
    hash) counts as `updated` rather than getting its own bucket.

    Files are skipped silently, not reported, in three cases: unreadable
    or non-UTF-8 bytes, whitespace-only content, and anything excluded by
    the collection's pattern or ignore rules. A fourth case is skipped but
    *counted*: files over `MAX_INDEXABLE_CHARS` (see `skipped_oversize`).
    A previously indexed file that has since crossed the cap is treated
    like any other skipped file - its document is deactivated.

    Args:
        name: Collection name, resolved against `user`'s own collections.

    Returns:
        Per-bucket counts; see `ReindexResult`.

    Raises:
        CollectionNotFoundError: No collection of that name owned by `user`.
        PermissionDeniedError: `can_access()` refused `write` on it.

    Note:
        Reads every matched file and hashes it on each run - there is no
        mtime shortcut. Content addressing makes that cheap in storage (an
        unchanged file writes nothing) but not in I/O, so the cost scales
        with total collection size, not with how much changed.
    """
    with log_duration(logger, f"reindex {name}") as timing:
        result = await _reindex_collection_impl(session, user, name)
        # Inside the block: log_duration emits the line on exit, so
        # fields added afterwards would never reach it.
        timing.update(
            {
                "indexed": result.indexed,
                "updated": result.updated,
                "unchanged": result.unchanged,
                "removed": result.removed,
                "skipped_oversize": result.skipped_oversize,
            }
        )
    return result


async def _reindex_collection_impl(
    session: AsyncSession, user: CurrentUser, name: str
) -> ReindexResult:
    collection = await _resolve_owned_collection(session, user, name, "write")
    files = _discover_files(collection.path, collection.pattern, collection.ignore_patterns)
    root = Path(collection.path)
    indexed = 0
    updated = 0
    unchanged = 0
    skipped_oversize = 0
    seen_paths: set[str] = set()

    for rel_path in files:
        filepath = root / rel_path
        try:
            content = filepath.read_text(encoding="utf-8")
            # Inside the try, not after it: a file deleted between the
            # read and the stat (a branch switch mid-walk - the exact
            # fast-moving-worktree case this indexer supports) must count
            # as skipped, not raise FileNotFoundError.
            mtime = filepath.stat().st_mtime
        except (OSError, UnicodeDecodeError) as exc:
            logger.warning("skipping %s: %s: %s", rel_path, type(exc).__name__, exc)
            continue
        if not content.strip():
            logger.warning("skipping %s: file is empty or whitespace-only", rel_path)
            continue
        if len(content) > MAX_INDEXABLE_CHARS:
            skipped_oversize += 1
            logger.warning(
                "skipping %s: %d chars exceeds the %d-char index limit",
                rel_path,
                len(content),
                MAX_INDEXABLE_CHARS,
            )
            continue
        seen_paths.add(rel_path)

        digest = hash_content(content)
        title = extract_title(content, rel_path)
        modified_at = datetime.fromtimestamp(mtime, tz=UTC)

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
        skipped_oversize=skipped_oversize,
    )
