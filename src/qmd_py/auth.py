"""ACL mock.

The user's real use case is a single local user, but the schema and call
sites are structured so wiring up real multi-user permission checks later
is additive, not a rearchitecture:

- `get_current_user()` ensures/returns one real `User` row (not just an
  abstract concept - `Collection.owner_user_id` is a real NOT NULL FK, so
  something concrete has to own every collection from the start).
- `can_access()` is *the* choke point every service-layer function calls
  before touching a collection's data (search, get, multi_get, collection
  management - see store.py). It's mocked to always return True regardless
  of `CollectionGrant` rows; swapping in a real check later only means
  changing this one function's body, not any of its call sites.
"""

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col

from qmd_py.config import get_settings
from qmd_py.db.models import Collection, User


@dataclass(frozen=True)
class CurrentUser:
    id: int
    email: str
    is_admin: bool


async def get_current_user(session: AsyncSession) -> CurrentUser:
    """Mock: one local user, created on first use if it doesn't exist yet.

    Real version: derive from an API key / session / OIDC token instead of
    a fixed settings-provided email.
    """
    email = get_settings().default_user_email
    stmt = (
        pg_insert(User)
        .values(email=email, is_admin=True)
        .on_conflict_do_nothing(index_elements=["email"])
    )
    await session.execute(stmt)
    await session.flush()

    result = await session.execute(select(User).where(col(User.email) == email))
    user = result.scalar_one()
    return CurrentUser(id=user.id, email=user.email, is_admin=user.is_admin)


async def can_access(
    user: CurrentUser, collection: Collection, permission: str = "read"
) -> bool:
    """ACL choke point - mocked to always allow.

    Real version: `user.is_admin or user.id == collection.owner_user_id or
    exists(CollectionGrant matching user/collection/permission)`.
    """
    del user, collection, permission  # unused until real checks land
    return True
