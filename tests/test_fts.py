"""Phase 4: build_ts_query unit tests (ported from the TS reference's
buildTsQuery docstring examples) + search_fts integration tests against a
real scratch Postgres schema (see conftest.py).
"""

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from qmd_py.auth import CurrentUser
from qmd_py.search.fts import build_ts_query, search_fts
from qmd_py.store import (
    add_collection,
    add_context,
    deactivate_document,
    insert_content,
    insert_document,
    utcnow,
)


def test_plain_terms_are_prefix_matched_and_anded() -> None:
    assert build_ts_query("performance sports") == "performance:* & sports:*"


def test_negation_uses_unary_not() -> None:
    assert build_ts_query("performance -sports") == "performance:* & !sports:*"


def test_quoted_phrase_becomes_adjacency() -> None:
    assert build_ts_query('"machine learning"') == "(machine <-> learning)"


def test_hyphenated_compound_becomes_phrase() -> None:
    assert build_ts_query("multi-agent memory") == "(multi <-> agent) & memory:*"


def test_dotted_token_is_prefix_matched_as_one_lexeme() -> None:
    assert build_ts_query("2026.4.10") == "2026.4.10:*"


def test_negative_only_query_returns_none() -> None:
    assert build_ts_query("-multi-agent") is None


def test_blank_query_returns_none() -> None:
    assert build_ts_query("") is None
    assert build_ts_query("the and or") == "the:* & and:* & or:*"


def test_cjk_run_becomes_character_adjacency_phrase() -> None:
    assert build_ts_query("文档") == "(文 <-> 档)"


def test_apostrophe_is_kept_in_term() -> None:
    assert build_ts_query("project's") == "project's:*"


@pytest.mark.integration
async def test_search_fts_finds_indexed_document(
    session: AsyncSession, user: CurrentUser
) -> None:
    collection = await add_collection(session, user, "docs", "/tmp/docs")
    await insert_content(session, "hash1", "This document explains error handling patterns.")
    await insert_document(
        session, collection.id, "errors.md", "Error Handling", "hash1", utcnow(), utcnow()
    )
    await session.commit()

    results = await search_fts(session, user, "error handling")
    assert len(results) == 1
    assert results[0].display_path == "docs/errors.md"
    assert results[0].docid == "hash1"[:6]
    assert results[0].source == "fts"
    assert 0.0 < results[0].score <= 1.0


@pytest.mark.integration
async def test_search_fts_respects_collection_filter(
    session: AsyncSession, user: CurrentUser
) -> None:
    coll_a = await add_collection(session, user, "coll-a", "/tmp/a")
    coll_b = await add_collection(session, user, "coll-b", "/tmp/b")
    await insert_content(session, "hasha", "shared keyword alpha")
    await insert_document(session, coll_a.id, "a.md", "A", "hasha", utcnow(), utcnow())
    await insert_content(session, "hashb", "shared keyword beta")
    await insert_document(session, coll_b.id, "b.md", "B", "hashb", utcnow(), utcnow())
    await session.commit()

    all_results = await search_fts(session, user, "shared keyword")
    assert {r.collection_name for r in all_results} == {"coll-a", "coll-b"}

    scoped = await search_fts(session, user, "shared keyword", collection_name="coll-a")
    assert [r.collection_name for r in scoped] == ["coll-a"]


@pytest.mark.integration
async def test_search_fts_includes_hierarchical_context(
    session: AsyncSession, user: CurrentUser
) -> None:
    collection = await add_collection(session, user, "notes", "/tmp/notes")
    await add_context(session, user, "notes", "", "root context")
    await add_context(session, user, "notes", "journal", "journal context")
    await insert_content(session, "hashc", "unique-token-42 appears here")
    await insert_document(
        session, collection.id, "journal/entry.md", "Entry", "hashc", utcnow(), utcnow()
    )
    await session.commit()

    results = await search_fts(session, user, "unique-token-42")
    assert len(results) == 1
    assert results[0].context == "root context\n\njournal context"


@pytest.mark.integration
async def test_search_fts_excludes_inactive_documents(
    session: AsyncSession, user: CurrentUser
) -> None:
    collection = await add_collection(session, user, "docs2", "/tmp/docs2")
    await insert_content(session, "hashd", "searchable-term-xyz content")
    await insert_document(session, collection.id, "gone.md", "Gone", "hashd", utcnow(), utcnow())
    await session.commit()

    assert len(await search_fts(session, user, "searchable-term-xyz")) == 1

    await deactivate_document(session, collection.id, "gone.md")
    await session.commit()
    assert await search_fts(session, user, "searchable-term-xyz") == []
