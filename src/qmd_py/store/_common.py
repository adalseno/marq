"""Shared primitives for the store package: the error types, the ACL
choke point every other module funnels collection lookups through, and
the small pure helpers (hashing, title extraction, line numbering).

Imports nothing from its sibling modules, so it can be the base of the
package's dependency order without cycles.
"""

import hashlib
import re
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col

from qmd_py.auth import CurrentUser, can_access
from qmd_py.db.models import Collection


class CollectionNotFoundError(Exception):
    pass


class PermissionDeniedError(Exception):
    pass


def hash_content(content: str) -> str:
    """SHA256 hex digest - matches the TS reference's `hashContent()`
    exactly (content-addressable hashes must agree byte-for-byte with the
    TS side for the Phase 3 parity check to mean anything).

    Sync: pure CPU, no I/O to await. It was `async` only because the TS
    original returns a Promise.
    """
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


async def can_read(user: CurrentUser, collection: Collection) -> bool:
    """The store package's one entry to `auth.can_access()` for the
    read-filter shape (`list_collections`), alongside
    `_resolve_owned_collection` below for the name-lookup shape.

    Exists so `can_access` is imported into exactly one module in this
    package. Before the package split it was one flat module, so
    tests/test_acl_gating.py could prove every choke point gates by
    patching a single name; spreading the import across submodules would
    have meant patching each one, and a new module importing it directly
    would silently escape that proof.
    """
    return await can_access(user, collection, "read")


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


def utcnow() -> datetime:
    return datetime.now(UTC)


def add_line_numbers(text: str, start_line: int = 1) -> str:
    lines = text.split("\n")
    return "\n".join(f"{start_line + i}: {line}" for i, line in enumerate(lines))
