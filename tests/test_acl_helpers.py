"""Phase 11: search/_acl.py helper tests not covered by test_acl_gating.py."""

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from qmd_py.auth import CurrentUser
from qmd_py.search._acl import collection_names_by_id
from qmd_py.store import add_collection

pytestmark = pytest.mark.integration


async def test_collection_names_by_id_empty_ids_returns_empty_dict(
    session: AsyncSession,
) -> None:
    assert await collection_names_by_id(session, []) == {}
    assert await collection_names_by_id(session, set()) == {}


async def test_collection_names_by_id_maps_ids_to_names(
    session: AsyncSession, user: CurrentUser
) -> None:
    collection = await add_collection(session, user, "notes", "/tmp/notes")
    await session.commit()

    names = await collection_names_by_id(session, [collection.id])
    assert names == {collection.id: "notes"}
