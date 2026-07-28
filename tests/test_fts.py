"""Phase 4: build_ts_query unit tests (ported from the TS reference's
buildTsQuery docstring examples) + search_fts integration tests against a
real scratch Postgres schema (see conftest.py).
"""

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from qmd_py.auth import CurrentUser
from qmd_py.search.fts import (
    build_ts_query,
    is_dotted_token,
    is_hyphenated_token,
    search_fts,
    update_document_search_vector,
)
from qmd_py.store import (
    add_collection,
    add_context,
    deactivate_document,
    insert_content,
    insert_document,
    set_global_context,
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


def test_interior_apostrophe_is_kept() -> None:
    assert build_ts_query("don't") == "don't:*"


def test_leading_apostrophe_is_stripped() -> None:
    """Regression: `'` is tsquery's lexeme-quote character, so a term
    starting with one (`'n:*`) is a tsquery syntax error - Postgres raised
    a raw ProgrammingError on `rock 'n roll` before the strip."""
    assert build_ts_query("rock 'n roll") == "rock:* & n:* & roll:*"


def test_all_apostrophe_terms_are_dropped() -> None:
    assert build_ts_query("'") is None
    assert build_ts_query("'' sports") == "sports:*"


def test_empty_quoted_phrase_is_dropped() -> None:
    assert build_ts_query('""') is None
    assert build_ts_query('"" sports') == "sports:*"


def test_unmatched_quote_consumes_the_rest_as_a_phrase() -> None:
    """No closing quote just runs to end of input - a soft degrade. The
    CLI/MCP surfaces reject this earlier via validate_lex_query."""
    assert build_ts_query('"machine learning') == "(machine <-> learning)"


def test_negated_quoted_phrase() -> None:
    assert build_ts_query('sports -"instant replay"') == "sports:* & !(instant <-> replay)"


def test_trailing_whitespace_is_ignored() -> None:
    assert build_ts_query("sports   ") == "sports:*"
    assert build_ts_query("   ") is None


def test_punctuation_only_terms_are_dropped() -> None:
    assert build_ts_query("--") is None
    assert build_ts_query("C++ &") == "c:*"


def test_leading_and_trailing_hyphens_are_not_compounds() -> None:
    """is_hyphenated_token requires a word character at both ends, so
    these fall through to ordinary prefix terms."""
    assert build_ts_query("foo-") == "foo:*"
    assert is_hyphenated_token("multi-agent") is True
    assert is_hyphenated_token("-agent") is False
    assert is_hyphenated_token("agent-") is False
    assert is_hyphenated_token("noseparator") is False


def test_dotted_token_classification() -> None:
    assert is_dotted_token("2026.4.10") is True
    assert is_dotted_token("v1.2") is True
    assert is_dotted_token("plain") is False
    assert is_dotted_token("trailing.") is False
    assert is_dotted_token(".leading") is False


def test_mixed_cjk_and_latin_term() -> None:
    assert build_ts_query("文档 report") == "(文 <-> 档) & report:*"


def test_negated_cjk_term() -> None:
    assert build_ts_query("report -文档") == "report:* & !(文 <-> 档)"


def test_multiple_negations_are_all_applied() -> None:
    assert build_ts_query("sports -baseball -tennis") == "sports:* & !baseball:* & !tennis:*"


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
async def test_search_fts_returns_empty_for_an_unparseable_query(
    session: AsyncSession, user: CurrentUser
) -> None:
    """A negative-only query has no positive term, so build_ts_query
    returns None and the search short-circuits to [] rather than issuing
    an invalid tsquery."""
    collection = await add_collection(session, user, "neg", "/tmp/neg")
    await insert_content(session, "hashneg", "some searchable body")
    await insert_document(
        session, collection.id, "n.md", "N", "hashneg", utcnow(), utcnow()
    )
    await session.commit()

    assert await search_fts(session, user, "-searchable") == []


@pytest.mark.integration
async def test_search_fts_survives_leading_apostrophe_terms(
    session: AsyncSession, user: CurrentUser
) -> None:
    """Regression: `rock 'n roll` used to reach Postgres as
    `rock:* & 'n:* & roll:*` - a tsquery syntax error (`'` opens a quoted
    lexeme) surfacing as an unhandled ProgrammingError."""
    collection = await add_collection(session, user, "apos", "/tmp/apos")
    await insert_content(session, "hashapos", "a history of rock n roll music")
    await insert_document(
        session, collection.id, "rock.md", "Rock", "hashapos", utcnow(), utcnow()
    )
    await session.commit()

    results = await search_fts(session, user, "rock 'n roll")
    assert [r.display_path for r in results] == ["apos/rock.md"]
    assert await search_fts(session, user, "'") == []


@pytest.mark.integration
async def test_context_includes_the_users_global_context(
    session: AsyncSession, user: CurrentUser
) -> None:
    """`context add /` applies across collections and comes first, before
    any per-path context."""
    collection = await add_collection(session, user, "glob", "/tmp/glob")
    await set_global_context(session, user, "global preamble")
    await add_context(session, user, "glob", "", "collection context")
    await insert_content(session, "hashg", "unique-global-token here")
    await insert_document(
        session, collection.id, "g.md", "G", "hashg", utcnow(), utcnow()
    )
    await session.commit()

    results = await search_fts(session, user, "unique-global-token")
    assert results[0].context == "global preamble\n\ncollection context"


@pytest.mark.integration
async def test_search_vector_is_cleared_when_a_document_goes_inactive(
    session: AsyncSession, user: CurrentUser
) -> None:
    """Recomputing the vector for a document that is no longer active
    nulls it out instead of leaving a stale one behind."""
    collection = await add_collection(session, user, "inact", "/tmp/inact")
    await insert_content(session, "hashi", "indexed body text")
    document = await insert_document(
        session, collection.id, "i.md", "I", "hashi", utcnow(), utcnow()
    )
    await session.commit()

    await deactivate_document(session, collection.id, "i.md")
    await update_document_search_vector(session, document.id)
    await session.commit()

    # Refresh explicitly: search_vector is written by raw UPDATE, so the
    # cached instance would otherwise lazy-load it as unexpected IO.
    await session.refresh(document)
    assert document.search_vector is None


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
