"""MCP server tests driven through a real `ClientSession` over the SDK's
in-memory transport, against a throwaway Postgres schema (conftest.py's
`mcp_env` fixture).

This is the protocol-level pass test_mcp.py's module docstring says is
missing: `initialize()`, `tools/list`, `tools/call`, and `resources/read`
all go through the actual JSON-RPC handlers rather than calling helpers
directly. The tool bodies open their own `get_session()`, which `mcp_env`
points at the test schema.

Covered here: `get`, `multi_get`, `status`, the `marq://` resource, and
`query`'s argument validation - everything that does not need the LLM
router. `query`'s search path itself needs a live router and is exercised
by test_hybrid.py.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import anyio
import httpx
import pytest
from mcp.client.session import ClientSession
from mcp.shared.memory import create_client_server_memory_streams
from mcp.types import CallToolResult, EmbeddedResource, TextContent, TextResourceContents

from qmd_py.auth import get_current_user
from qmd_py.db.engine import get_session
from qmd_py.mcp.server import create_mcp_server
from qmd_py.store import add_collection, add_context, insert_content, insert_document, utcnow

pytestmark = pytest.mark.integration


async def _seed_documents() -> None:
    async with get_session() as session:
        user = await get_current_user(session)
        collection = await add_collection(session, user, "docs", "/tmp/docs", "**/*.md")
        await insert_content(session, "aaaaaa1111", "# Alpha\n\nunique-alpha-token here\n")
        await insert_document(
            session, collection.id, "notes/alpha.md", "Alpha", "aaaaaa1111", utcnow(), utcnow()
        )
        await insert_content(session, "bbbbbb2222", "# Beta\n\nunique-beta-token there\n")
        await insert_document(
            session, collection.id, "notes/beta.md", "Beta", "bbbbbb2222", utcnow(), utcnow()
        )
        await session.commit()


@asynccontextmanager
async def _client() -> AsyncIterator[ClientSession]:
    """A `ClientSession` wired to a real server over in-memory streams.

    The SDK used to ship this as
    `mcp.shared.memory.create_connected_server_and_client_session`, which
    2.0 removed; only the stream pair below survives, so the plumbing it
    used to hide - run the server in a task group, hand the other ends to
    the client - lives here now. The session is deliberately *not*
    initialized: the tests drive `initialize()` themselves, since its
    response (serverInfo, instructions) is itself under test.
    """
    server = await create_mcp_server()
    low = server._lowlevel_server
    async with create_client_server_memory_streams() as (
        (client_read, client_write),
        (server_read, server_write),
    ):
        async with anyio.create_task_group() as tg:
            tg.start_soon(
                lambda: low.run(
                    server_read,
                    server_write,
                    low.create_initialization_options(),
                    raise_exceptions=True,
                )
            )
            async with ClientSession(client_read, client_write) as session:
                yield session
            # Nothing else stops the server task, and leaving it running
            # would hang the surrounding task group on exit.
            tg.cancel_scope.cancel()


def _text_of(result: CallToolResult) -> str:
    """Flattens a CallToolResult's content blocks into one string, whether
    they arrived as plain text or as embedded document resources."""
    chunks = []
    for block in result.content:
        if isinstance(block, TextContent):
            chunks.append(block.text)
        elif isinstance(block, EmbeddedResource) and isinstance(
            block.resource, TextResourceContents
        ):
            chunks.append(block.resource.text)
    return "\n".join(chunks)


async def test_initialize_reports_marq_server_and_own_version(mcp_env: str) -> None:
    """Regression guard on the server's reported version: left to itself
    the SDK reports *its own* version here, not qmd-py's."""
    from importlib.metadata import version

    await _seed_documents()
    async with _client() as session:
        result = await session.initialize()

    assert result.server_info.name == "marq"
    assert result.server_info.version == version("qmd-py")
    assert result.instructions is not None
    assert "2 documents" in result.instructions
    assert "docs" in result.instructions


async def test_list_tools_exposes_the_four_documented_tools(mcp_env: str) -> None:
    await _seed_documents()
    async with _client() as session:
        await session.initialize()
        tools = await session.list_tools()

    assert {t.name for t in tools.tools} == {"query", "get", "multi_get", "status"}


async def test_get_tool_returns_document_as_embedded_resource(mcp_env: str) -> None:
    await _seed_documents()
    async with _client() as session:
        await session.initialize()
        result = await session.call_tool("get", {"file": "notes/alpha.md"})

    assert result.is_error is not True
    assert "1: # Alpha" in _text_of(result)


async def test_get_tool_honours_line_range_suffix(mcp_env: str) -> None:
    await _seed_documents()
    async with _client() as session:
        await session.initialize()
        result = await session.call_tool("get", {"file": "notes/alpha.md:3:1"})

    text = _text_of(result)
    assert "3: unique-alpha-token here" in text
    assert "# Alpha" not in text


async def test_get_tool_without_line_numbers(mcp_env: str) -> None:
    await _seed_documents()
    async with _client() as session:
        await session.initialize()
        result = await session.call_tool(
            "get", {"file": "notes/alpha.md", "lineNumbers": False}
        )

    text = _text_of(result)
    assert "# Alpha" in text
    assert "1: # Alpha" not in text


async def test_get_tool_missing_document_is_an_error_with_suggestions(mcp_env: str) -> None:
    await _seed_documents()
    async with _client() as session:
        await session.initialize()
        result = await session.call_tool("get", {"file": "notes/alpho.md"})

    assert result.is_error is True
    text = _text_of(result)
    assert "Document not found" in text
    assert "notes/alpha.md" in text


async def test_get_tool_prepends_folder_context(mcp_env: str) -> None:
    await _seed_documents()
    async with get_session() as session:
        user = await get_current_user(session)
        await add_context(session, user, "docs", "notes", "these are my notes")
        await session.commit()

    async with _client() as session:
        await session.initialize()
        result = await session.call_tool("get", {"file": "notes/alpha.md"})

    assert "<!-- Context: these are my notes -->" in _text_of(result)


async def test_multi_get_tool_matches_glob(mcp_env: str) -> None:
    await _seed_documents()
    async with _client() as session:
        await session.initialize()
        result = await session.call_tool("multi_get", {"pattern": "notes/*.md"})

    text = _text_of(result)
    assert result.is_error is not True
    assert "unique-alpha-token" in text
    assert "unique-beta-token" in text


async def test_multi_get_tool_skips_oversized_files(mcp_env: str) -> None:
    await _seed_documents()
    async with _client() as session:
        await session.initialize()
        result = await session.call_tool(
            "multi_get", {"pattern": "notes/*.md", "maxBytes": 10}
        )

    text = _text_of(result)
    assert "[SKIPPED:" in text
    assert "unique-alpha-token" not in text


async def test_multi_get_tool_without_match_is_an_error(mcp_env: str) -> None:
    await _seed_documents()
    async with _client() as session:
        await session.initialize()
        result = await session.call_tool("multi_get", {"pattern": "nothing/*.md"})

    assert result.is_error is True
    assert "No files matched pattern" in _text_of(result)


async def test_status_tool_reports_counts_and_structured_content(mcp_env: str) -> None:
    await _seed_documents()
    async with _client() as session:
        await session.initialize()
        result = await session.call_tool("status", {})

    assert "Total documents: 2" in _text_of(result)
    structured = result.structured_content
    assert structured is not None
    assert structured["totalDocuments"] == 2
    assert structured["hasVectorIndex"] is False
    assert [c["name"] for c in structured["collections"]] == ["docs"]


async def test_query_tool_rejects_missing_arguments(mcp_env: str) -> None:
    """Both validation branches return before any LLM call, so they're
    reachable without a live router."""
    await _seed_documents()
    async with _client() as session:
        await session.initialize()
        result = await session.call_tool("query", {})

    assert result.is_error is True
    assert "provide either 'query'" in _text_of(result)


async def test_query_tool_rejects_mutually_exclusive_arguments(mcp_env: str) -> None:
    await _seed_documents()
    async with _client() as session:
        await session.initialize()
        result = await session.call_tool(
            "query",
            {"query": "alpha", "searches": [{"type": "lex", "query": "alpha"}]},
        )

    assert result.is_error is True
    assert "mutually exclusive" in _text_of(result)


async def test_query_tool_rejects_negation_in_a_vec_sub_query(mcp_env: str) -> None:
    """Caller-supplied sub-queries are validated before any LLM call."""
    await _seed_documents()
    async with _client() as session:
        await session.initialize()
        result = await session.call_tool(
            "query", {"searches": [{"type": "vec", "query": "sports -baseball"}]}
        )

    assert result.is_error is True
    text = _text_of(result)
    assert "vec: Negation (-term) is not supported" in text


async def test_query_tool_rejects_multiline_lex_sub_query(mcp_env: str) -> None:
    """A newline can't reach a lex sub-query through the CLI's line-split
    syntax, but an MCP client can send one directly."""
    await _seed_documents()
    async with _client() as session:
        await session.initialize()
        result = await session.call_tool(
            "query", {"searches": [{"type": "lex", "query": "alpha\nbeta"}]}
        )

    assert result.is_error is True
    assert "must be a single line" in _text_of(result)


async def test_document_resource_reads_slash_spanning_path(mcp_env: str) -> None:
    """The whole reason the resource bypasses MCPServer's @mcp.resource()
    decorator: its template matching can't express a path segment
    containing slashes."""
    await _seed_documents()
    async with _client() as session:
        await session.initialize()
        result = await session.read_resource("marq://docs/notes/alpha.md")

    assert len(result.contents) == 1
    contents = result.contents[0]
    assert isinstance(contents, TextResourceContents)
    assert "1: # Alpha" in contents.text


async def test_document_resource_percent_decodes_path(mcp_env: str) -> None:
    async with get_session() as session:
        user = await get_current_user(session)
        collection = await add_collection(session, user, "docs", "/tmp/docs", "**/*.md")
        await insert_content(session, "cccccc3333", "# Spaced\n\nspaced body\n")
        await insert_document(
            session, collection.id, "my file.md", "Spaced", "cccccc3333", utcnow(), utcnow()
        )
        await session.commit()

    async with _client() as session:
        await session.initialize()
        result = await session.read_resource("marq://docs/my%20file.md")

    contents = result.contents[0]
    assert isinstance(contents, TextResourceContents)
    assert "spaced body" in contents.text


async def test_document_resource_missing_document_reports_not_found(mcp_env: str) -> None:
    await _seed_documents()
    async with _client() as session:
        await session.initialize()
        result = await session.read_resource("marq://docs/notes/ghost.md")

    contents = result.contents[0]
    assert isinstance(contents, TextResourceContents)
    assert "Document not found" in contents.text


async def test_rest_query_ignores_json_booleans_in_integer_params(mcp_env: str) -> None:
    """Regression for the third review's finding 8: Python's bool
    subclasses int, so `"limit": true` passed the isinstance(int) guard
    and quietly ran with limit 1. A boolean must fall back to the
    default instead of masquerading as an integer."""
    await _seed_documents()
    server = await create_mcp_server(http=True)
    transport = httpx.ASGITransport(app=server.streamable_http_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/query",
            json={
                "searches": [{"type": "lex", "query": "unique"}],
                "limit": True,
                "rerank": False,
            },
        )

    assert response.status_code == 200
    # Both seeded documents match; a limit of true-as-1 would return one.
    assert len(response.json()["results"]) == 2


async def test_rest_query_clamps_a_negative_limit(mcp_env: str) -> None:
    """A negative limit passed the `_is_int` fence and reached
    `results[:limit]` as `results[:-1]`, silently dropping results off the
    *end* - the caller asked for fewer and got "all but the last". Clamped
    to zero, so a nonsensical limit yields nothing rather than a
    plausible-looking truncation."""
    await _seed_documents()
    server = await create_mcp_server(http=True)
    transport = httpx.ASGITransport(app=server.streamable_http_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/query",
            json={
                "searches": [{"type": "lex", "query": "unique"}],
                "limit": -1,
                "rerank": False,
            },
        )

    assert response.status_code == 200
    # Both seeded documents match; results[:-1] would have returned one.
    assert response.json()["results"] == []


async def test_rest_query_rejects_malformed_searches_with_400(mcp_env: str) -> None:
    """Regression: a non-dict `searches` entry (AttributeError) or an
    invalid `type` value (pydantic ValidationError) used to escape the
    handler and surface as a 500, unlike every other malformed-input path
    in the same route."""
    server = await create_mcp_server(http=True)
    transport = httpx.ASGITransport(app=server.streamable_http_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        for body in (
            {"searches": ["foo"]},
            {"searches": [{"type": "nope", "query": "x"}]},
            {"searches": [None]},
            {"searches": "not-a-list"},
            {},
        ):
            response = await client.post("/query", json=body)
            assert response.status_code == 400, body
            assert "error" in response.json()
