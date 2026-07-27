"""Shared collection/ACL resolution for the search submodule (fts.py,
vector.py). Deliberately does not import from `store.py` - see fts.py's
module docstring for why (store.py depends on this package, so this
package must not depend back on store.py).
"""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col

from qmd_py.auth import CurrentUser, can_access
from qmd_py.db.models import Collection


async def resolve_collection_ids(
    session: AsyncSession, user: CurrentUser, collection_name: str | None
) -> list[int]:
    """Collection ids `user` can read, optionally narrowed to one name. No
    name match / no accessible collections -> empty list (callers should
    treat that as "no results", not an error - matches the TS reference's
    searchFTS/searchVec, which never validate the collection filter, they
    just produce zero matching rows)."""
    result = await session.execute(select(Collection).order_by(col(Collection.name)))
    collections = [c for c in result.scalars() if await can_access(user, c, "read")]
    if collection_name is not None:
        collections = [c for c in collections if c.name == collection_name]
    return [c.id for c in collections]


async def collection_names_by_id(
    session: AsyncSession, ids: set[int] | list[int]
) -> dict[int, str]:
    """Look up collection names in bulk, for labelling result rows.

    Deliberately unfiltered by ACL: callers pass ids they already
    resolved through `resolve_collection_ids()`, so re-checking would be
    redundant.

    Returns:
        An id-to-name mapping; empty for empty input, without querying.
    """
    if not ids:
        return {}
    result = await session.execute(
        select(col(Collection.id), col(Collection.name)).where(col(Collection.id).in_(ids))
    )
    return {row.id: row.name for row in result}
