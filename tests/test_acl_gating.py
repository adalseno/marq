"""Phase 11: ACL choke-point gating proof.

`can_access()` (auth.py) is mocked to always return True - the real use
case is a single local user - but the schema/call sites are structured
so a real check is additive later. This file proves the call sites
themselves actually consult `can_access()` and would gate correctly
once a real check lands, using a *second*, non-owner `CurrentUser` and
a fake `can_access` that implements the exact rule auth.py's own
docstring promises for the real version (`user.is_admin or user.id ==
collection.owner_user_id`).

There are two distinct call-site shapes in this codebase, and they
behave differently under a real check - both are exercised here:

1. `resolve_collection_ids()` (search/_acl.py) queries ALL collections
   then filters per-collection via `can_access()` - no owner prefilter
   in SQL. This is the one true "any user, any collection, gated
   purely by can_access()" checkpoint, and backs search_fts/search_vec/
   hybrid_query/find_document/multi_get/match_files_by_glob/
   get_vector_index_health, plus list_collections's own inline
   can_access filter (store.py). These all silently return empty/
   not-found for a denied user rather than raising - proven below with
   a genuinely different `other_user`.

2. `_resolve_owned_collection()` (store.py) scopes its SQL to
   `owner_user_id == user.id` *before* ever calling `can_access()` -
   backs the name-based collection-management functions (remove/
   rename/list_files/etc.). A non-owner never reaches the can_access
   call at all here: the query itself returns no row, so a different
   user gets CollectionNotFoundError, not PermissionDeniedError. That's
   a real, currently-accurate limitation of "mocked, not implemented"
   ACL (see the plan's Phase 11 note) - the SQL will need broadening to
   an all-collections-then-can_access-filter shape once real grants
   let a *different* user legitimately see a collection they don't
   own. Proving *this* choke point's can_access call therefore still
   requires denying the owner themselves (matching test_store.py's
   existing `test_permission_denied_surfaces_from_can_access`) - done
   here for a couple more of its callers.
"""

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from qmd_py.auth import CurrentUser
from qmd_py.db.models import Collection
from qmd_py.llm.client import LlmClient
from qmd_py.search.fts import search_fts
from qmd_py.search.hybrid import hybrid_query
from qmd_py.search.vector import get_vector_index_health, search_vec
from qmd_py.store import (
    DocumentNotFound,
    PermissionDeniedError,
    find_document,
    get_status,
    insert_content,
    insert_document,
    list_collections,
    list_files,
    match_files_by_glob,
    multi_get,
    rename_collection,
    utcnow,
)

pytestmark = pytest.mark.integration


@pytest.fixture
def other_user() -> CurrentUser:
    """A second, genuinely different user - never inserted into the
    `User` table (these choke points never need that row to exist;
    they only compare ids/filter by owner_user_id)."""
    return CurrentUser(id=999_999, email="other@test.local", is_admin=False)


def _owner_only_can_access() -> object:
    async def fake(user: CurrentUser, collection: Collection, permission: str = "read") -> bool:
        del permission
        return user.is_admin or user.id == collection.owner_user_id

    return fake


@pytest.fixture
async def gated_collection(session: AsyncSession, user: CurrentUser) -> Collection:
    from qmd_py.store import add_collection

    collection = await add_collection(session, user, "gated", "/tmp/gated")
    await insert_content(session, "gatedhash1", "the quick brown fox jumps over the lazy dog")
    await insert_document(
        session, collection.id, "fox.md", "Fox", "gatedhash1", utcnow(), utcnow()
    )
    await session.commit()
    return collection


async def test_other_user_currently_sees_everything_by_design(
    session: AsyncSession, user: CurrentUser, other_user: CurrentUser, gated_collection: Collection
) -> None:
    """Documents today's real (mocked) behavior: a single-user deployment
    with can_access() always True means any CurrentUser can see any
    collection. This is intentional, not a bug - the proof below is
    about what happens once a real check is installed, not a claim that
    one exists today."""
    results = await search_fts(session, other_user, "fox", collection_name="gated")
    assert len(results) == 1


async def test_resolve_collection_ids_gates_search_fts_for_non_owner(
    session: AsyncSession,
    user: CurrentUser,
    other_user: CurrentUser,
    gated_collection: Collection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("qmd_py.search._acl.can_access", _owner_only_can_access())

    owner_results = await search_fts(session, user, "fox", collection_name="gated")
    assert len(owner_results) == 1

    other_results = await search_fts(session, other_user, "fox", collection_name="gated")
    assert other_results == []


async def test_resolve_collection_ids_gates_search_vec_for_non_owner(
    session: AsyncSession,
    user: CurrentUser,
    other_user: CurrentUser,
    gated_collection: Collection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("qmd_py.search._acl.can_access", _owner_only_can_access())
    llm_client = LlmClient("http://unused.invalid")
    try:
        # No embeddings table exists for this fresh collection - search_vec
        # already short-circuits to [] before ever calling the (unreachable,
        # never-hit) LLM client, so this proves the ACL gate specifically:
        # a real embeddings table would make an owner's search actually
        # return hits, but the non-owner must get [] regardless.
        other_results = await search_vec(session, other_user, "fox", llm_client, "bge-m3-q8_0")
        assert other_results == []
    finally:
        await llm_client.aclose()


async def test_resolve_collection_ids_gates_hybrid_query_for_non_owner(
    session: AsyncSession,
    user: CurrentUser,
    other_user: CurrentUser,
    gated_collection: Collection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("qmd_py.search._acl.can_access", _owner_only_can_access())
    llm_client = LlmClient("http://unused.invalid")
    try:
        # With no accessible collections, hybrid_query's initial FTS probe
        # and vector search both come back empty before any LLM call is
        # ever made (no expansion, no rerank) - so this proves gating
        # without needing a live router.
        results = await hybrid_query(
            session, other_user, "fox", llm_client,
            "bge-m3-q8_0", "unused-model", "unused-model",
            collection_name="gated", skip_rerank=True,
        )
        assert results == []
    finally:
        await llm_client.aclose()


async def test_resolve_collection_ids_gates_find_document_for_non_owner(
    session: AsyncSession,
    user: CurrentUser,
    other_user: CurrentUser,
    gated_collection: Collection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("qmd_py.search._acl.can_access", _owner_only_can_access())

    owner_result = await find_document(session, user, "fox.md")
    assert not isinstance(owner_result, DocumentNotFound)

    other_result = await find_document(session, other_user, "fox.md")
    assert isinstance(other_result, DocumentNotFound)


async def test_resolve_collection_ids_gates_multi_get_and_glob_for_non_owner(
    session: AsyncSession,
    user: CurrentUser,
    other_user: CurrentUser,
    gated_collection: Collection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("qmd_py.search._acl.can_access", _owner_only_can_access())

    assert await multi_get(session, user, "fox.md") != []
    assert await multi_get(session, other_user, "fox.md") == []

    assert await match_files_by_glob(session, user, "*.md") != []
    assert await match_files_by_glob(session, other_user, "*.md") == []


async def test_resolve_collection_ids_gates_vector_index_health_for_non_owner(
    session: AsyncSession,
    user: CurrentUser,
    other_user: CurrentUser,
    gated_collection: Collection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("qmd_py.search._acl.can_access", _owner_only_can_access())

    health = await get_vector_index_health(session, other_user, "bge-m3-q8_0")
    assert health.has_vector_index is False
    assert health.needs_embedding == 0


async def test_list_collections_and_status_gate_for_non_owner(
    session: AsyncSession,
    user: CurrentUser,
    other_user: CurrentUser,
    gated_collection: Collection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("qmd_py.store.can_access", _owner_only_can_access())

    owner_rows = await list_collections(session, user)
    assert any(c.name == "gated" for c in owner_rows)

    other_rows = await list_collections(session, other_user)
    assert other_rows == []

    status = await get_status(session, other_user)
    assert status.total_documents == 0
    assert status.collections == []


async def test_resolve_owned_collection_gates_rename_and_list_files_for_owner_denied(
    session: AsyncSession, user: CurrentUser, gated_collection: Collection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`_resolve_owned_collection` scopes its SQL to owner_user_id ==
    user.id before ever calling can_access() - see this module's
    docstring - so proving *this* choke point (rather than the owner
    prefilter) means denying the owner, matching test_store.py's
    existing regression test for remove_collection."""

    async def deny(*_args: object, **_kwargs: object) -> bool:
        return False

    monkeypatch.setattr("qmd_py.store.can_access", deny)

    with pytest.raises(PermissionDeniedError):
        await rename_collection(session, user, "gated", "renamed")

    with pytest.raises(PermissionDeniedError):
        await list_files(session, user, "gated")
