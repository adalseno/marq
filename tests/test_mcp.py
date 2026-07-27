"""Phase 9: MCP server unit tests - the pure/session-taking helpers
behind mcp/server.py's tools and dynamic instructions.

Full protocol-level exercise (a real MCP ClientSession driving `query`/
`get`/`multi_get`/`status` and the `qmd://` resource end to end) is
deliberately NOT pytested here: `create_mcp_server()`'s tool handlers
each open their own session via `db.engine.get_session()`, which is
bound to whatever the environment's real settings point at (the dev
container, per .env) - exactly like the CLI's `cli/runtime.py`. That's
the same reason this project's CLI commands (Phases 6-8) are manually
live-verified against the dev container and the real ubuserver.internal
server rather than pytested end-to-end; the MCP server follows the same
established split.
"""

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from qmd_py.auth import CurrentUser
from qmd_py.mcp.server import (
    _default_collection_names,
    _encode_qmd_path,
    _filter_by_collections,
    _format_search_summary,
    _parse_get_lookup,
    _round2,
    build_instructions,
)
from qmd_py.store import add_collection, insert_content, insert_document, utcnow


def test_encode_qmd_path_preserves_slashes_encodes_segments() -> None:
    assert _encode_qmd_path("docs/my file.md") == "docs/my%20file.md"


def test_round2_rounds_to_two_decimals() -> None:
    assert _round2(0.123456) == 0.12


def test_format_search_summary_empty() -> None:
    assert _format_search_summary([], "foo") == 'No results found for "foo"'


def test_format_search_summary_lists_results() -> None:
    items = [
        {"docid": "#abc123", "file": "a.md", "title": "A", "score": 0.856},
        {"docid": "#def456", "file": "b.md", "title": "B", "score": 0.5},
    ]
    summary = _format_search_summary(items, "foo")
    assert 'Found 2 results for "foo"' in summary
    assert "#abc123 86% a.md - A" in summary
    assert "#def456 50% b.md - B" in summary


def test_filter_by_collections_noop_for_single_collection() -> None:
    class R:
        def __init__(self, file: str) -> None:
            self.file = file

    results = [R("qmd://a/x.md")]
    assert _filter_by_collections(results, ["a"]) == results


def test_filter_by_collections_filters_for_multiple() -> None:
    class R:
        def __init__(self, file: str) -> None:
            self.file = file

    results = [R("qmd://a/x.md"), R("qmd://b/y.md"), R("qmd://c/z.md")]
    filtered = _filter_by_collections(results, ["a", "b"])
    assert [r.file for r in filtered] == ["qmd://a/x.md", "qmd://b/y.md"]


@pytest.mark.parametrize(
    ("file", "expected_lookup", "expected_from", "expected_max"),
    [
        ("foo.md", "foo.md", None, None),
        ("foo.md:100", "foo.md", 100, None),
        ("foo.md:100:40", "foo.md", 100, 40),
        ("#abc123:5:10", "#abc123", 5, 10),
    ],
)
def test_parse_get_lookup(
    file: str, expected_lookup: str, expected_from: int | None, expected_max: int | None
) -> None:
    lookup, from_line, max_lines = _parse_get_lookup(file, None, None)
    assert lookup == expected_lookup
    assert from_line == expected_from
    assert max_lines == expected_max


def test_parse_get_lookup_explicit_args_take_precedence() -> None:
    lookup, from_line, max_lines = _parse_get_lookup("foo.md:100:40", 5, 6)
    assert lookup == "foo.md"
    assert from_line == 5
    assert max_lines == 6


def test_parse_get_lookup_clamps_from_line_to_one() -> None:
    _, from_line, _ = _parse_get_lookup("foo.md", -3, None)
    assert from_line == 1


@pytest.mark.integration
async def test_default_collection_names_only_include_by_default(
    session: AsyncSession, user: CurrentUser
) -> None:
    await add_collection(session, user, "included", "/tmp/included")
    excluded = await add_collection(session, user, "excluded", "/tmp/excluded")
    excluded.include_by_default = False
    session.add(excluded)
    await session.commit()

    names = await _default_collection_names(session, user)
    assert names == ["included"]


@pytest.mark.integration
async def test_build_instructions_mentions_doc_count_and_collections(
    session: AsyncSession, user: CurrentUser
) -> None:
    collection = await add_collection(session, user, "notes", "/tmp/notes")
    await insert_content(session, "hash1", "hello world")
    await insert_document(session, collection.id, "a.md", "A", "hash1", utcnow(), utcnow())
    await session.commit()

    instructions = await build_instructions(session, user, "bge-m3-q8_0")
    assert "1 markdown documents" in instructions
    assert "notes" in instructions
    assert "No vector embeddings yet" in instructions
    assert "Search: Use `query`" in instructions
