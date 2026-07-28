"""MCP server: FastMCP tools (`query`/`get`/`multi_get`/`status`) plus a
raw `marq://{path}` document resource - port of the TS reference's
src/mcp/server.ts.

Each tool/resource handler opens its own short-lived DB session (see
cli/runtime.py's per-command `with_session_and_user` pattern) rather than
holding one session for the server's whole lifetime, since MCP tool calls
are concurrent and long-lived; the async engine's connection pool makes
this cheap.

The `document` resource is registered directly against the low-level
`mcp._mcp_server` (bypassing FastMCP's own `@mcp.resource()` decorator)
because that decorator's template matching only supports a single path
segment per `{param}` (regex `[^/]+`) - it cannot express the TS
reference's `qmd://{+path}` (RFC 6570 reserved expansion, matches
slashes). The low-level `read_resource()` handler receives the full raw
URI string instead, which we parse ourselves. No `list_resources` handler
is registered, matching the TS reference's `list: undefined` (documents
are discovered via the search tools, not listed).

Field names on tool input schemas deliberately stay camelCase
(`minScore`, `candidateLimit`, ...) rather than idiomatic snake_case:
this is a wire-compatibility surface for MCP clients, frozen to match the
TS reference's actual JSON schema, not internal Python code.
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from importlib.metadata import version
from typing import Annotated, Any, Literal
from urllib.parse import quote, unquote

from mcp.server.fastmcp import FastMCP
from mcp.server.lowlevel.helper_types import ReadResourceContents
from mcp.types import (
    AudioContent,
    CallToolResult,
    EmbeddedResource,
    ImageContent,
    ResourceLink,
    TextContent,
    TextResourceContents,
    ToolAnnotations,
)
from pydantic import BaseModel, Field, ValidationError

from qmd_py.auth import CurrentUser, get_current_user
from qmd_py.cli.snippet import extract_snippet
from qmd_py.config import get_settings
from qmd_py.db.engine import get_session
from qmd_py.llm.client import LlmClient
from qmd_py.log import setup_logging
from qmd_py.search.hybrid import (
    ExpandedQuery,
    ModelConfig,
    QueryOptions,
    hybrid_query,
    validate_typed_queries,
)
from qmd_py.search.vector import get_vector_index_health
from qmd_py.store import (
    DEFAULT_MULTI_GET_MAX_BYTES,
    DocumentDetail,
    DocumentNotFound,
    add_line_numbers,
    find_document,
    get_global_context,
    get_status,
    list_collections,
    multi_get,
)

logger = logging.getLogger(__name__)


def _encode_qmd_path(path: str) -> str:
    """Percent-encodes each path segment but preserves slashes, matching
    the TS reference's `encodeQmdPath`."""
    return "/".join(quote(segment, safe="") for segment in path.split("/"))


async def _default_collection_names(session: Any, user: CurrentUser) -> list[str]:
    rows = await list_collections(session, user)
    return [c.name for c in rows if c.include_by_default]


def _filter_by_collections(results: list[Any], names: list[str]) -> list[Any]:
    """Empty name list means nothing is in scope, not "no filter" - see
    cli/commands/read.py's `_filter_by_collections` for the full
    reasoning."""
    if not names:
        return []
    if len(names) == 1:
        return results
    prefixes = tuple(f"marq://{n}/" for n in names)
    return [r for r in results if r.file.startswith(prefixes)]


# =============================================================================
# Dynamic server instructions - built from actual index state, injected
# into the MCP `initialize` response so the LLM has immediate context
# without an extra tool call.
# =============================================================================


async def build_instructions(session: Any, user: CurrentUser, embed_model: str) -> str:
    """Compose the server instructions sent in the `initialize` response.

    Built from live index state - document counts, collection names,
    whether anything is embedded yet - so a client knows what is
    searchable without spending a tool call to find out.

    Returns:
        Markdown-ish plain text. Includes a prompt to run `marq embed`
        when embeddings are missing or stale.
    """
    status = await get_status(session, user)
    global_ctx = await get_global_context(session, user)
    health = await get_vector_index_health(session, user, embed_model)

    lines = [f"marq is your local search engine over {status.total_documents} documents."]
    if global_ctx:
        lines.append(f"Context: {global_ctx}")

    if status.collections:
        lines.append("")
        names = ", ".join(c.name for c in status.collections)
        lines.append(f"Collections (scope with `collections` parameter): {names}")
        lines.append(
            "Call the `status` tool for collection descriptions, paths, and "
            "per-collection doc counts."
        )

    if not health.has_vector_index:
        lines.append("")
        lines.append(
            "Note: No vector embeddings yet. Run `marq embed` to enable semantic search "
            "(vec/hyde)."
        )
    elif health.needs_embedding > 0:
        lines.append("")
        lines.append(f"Note: {health.needs_embedding} documents need embedding. "
                      "Run `marq embed` to update.")

    lines.append("")
    lines.append("Search: Use `query` with sub-queries (lex/vec/hyde):")
    lines.append("  - type:'lex' — BM25 keyword search (exact terms, fast)")
    lines.append("  - type:'vec' — semantic vector search (meaning-based)")
    lines.append("  - type:'hyde' — hypothetical document (write what the answer looks like)")
    lines.append("")
    lines.append("  Always provide `intent` on every search call to disambiguate and improve "
                  "snippets.")
    lines.append("")
    lines.append("Examples:")
    lines.append("  Quick keyword lookup: [{type:'lex', query:'error handling'}]")
    lines.append("  Semantic search: [{type:'vec', query:'how to handle errors gracefully'}]")
    lines.append(
        "  Best results: [{type:'lex', query:'error'}, "
        "{type:'vec', query:'error handling best practices'}]"
    )
    lines.append("  With intent: searches=[{type:'lex', query:'performance'}], "
                  "intent='web page load times'")
    lines.append("")
    lines.append("Retrieval:")
    lines.append(
        "  - `get` — single document by path or docid (#abc123). Supports a line-range "
        "suffix: `file.md:100` (from line 100) or `file.md:100:40` (40 lines from line 100)."
    )
    lines.append(
        "  - `multi_get` — batch retrieve by glob (`journals/2025-05*.md`) or "
        "comma-separated list."
    )
    lines.append("")
    lines.append("Tips:")
    lines.append("  - File paths in results are relative to their collection.")
    lines.append("  - Use `minScore: 0.5` to filter low-confidence results.")
    lines.append("  - Results include a `context` field describing the content type.")
    return "\n".join(lines)


# =============================================================================
# Tool: query
# =============================================================================


class SubSearch(BaseModel):
    """One typed sub-query in the `query` tool's `searches` argument.

    The wire-facing counterpart of `ExpandedQuery`; a pydantic model
    because its field descriptions become the JSON schema MCP clients
    read.
    """

    type: Literal["lex", "vec", "hyde"] = Field(
        description="lex = BM25 keywords (supports \"phrase\" and -negation); "
        "vec = semantic question; hyde = hypothetical answer passage"
    )
    query: str = Field(
        description="The query text. For lex: use keywords, \"quoted phrases\", and "
        "-negation. For vec: natural language question. For hyde: 50-100 word answer passage."
    )


def _round2(value: float) -> float:
    return round(value, 2)


async def _run_query_search(
    *,
    llm_client: LlmClient,
    query: str | None,
    searches: list[SubSearch] | None,
    limit: int,
    min_score: float,
    candidate_limit: int | None,
    collections: list[str] | None,
    intent: str | None,
    rerank: bool,
) -> tuple[list[dict[str, Any]], str]:
    """Shared core behind the `query` tool - runs hybrid_query() either in
    plain-text (auto-expanded) mode or structured (caller-supplied
    lex/vec/hyde sub-queries) mode, and shapes results into the TS
    reference's `SearchResultItem` dict shape. Returns (items, primary_query).

    `llm_client` is the server-lifetime client created in
    `create_mcp_server` - constructing one per call made every query pay
    TCP/TLS setup and pool warmup instead of reusing keep-alive
    connections (`httpx.AsyncClient` is concurrency-safe, which the
    gathered /tokenize fan-out already relies on)."""
    settings = get_settings()
    async with get_session() as session:
        user = await get_current_user(session)
        await session.commit()

        effective_collections = collections if collections else await _default_collection_names(
            session, user
        )
        single = (
            effective_collections[0] if len(effective_collections) == 1 else None
        )

        preexpanded = None
        primary_query = query or ""
        if searches:
            preexpanded = [ExpandedQuery(s.type, s.query) for s in searches]
            primary_query = (
                next((s.query for s in searches if s.type == "lex"), None)
                or next((s.query for s in searches if s.type == "vec"), None)
                or searches[0].query
            )

        # More than one collection means post-filtering by prefix below, so
        # fetch extra headroom the same way the CLI's search/vsearch/query
        # commands do (see cli/commands/read.py's _resolve_collection_names
        # callers).
        fetch_limit = limit if len(effective_collections) <= 1 else max(50, limit * 2)

        results = await hybrid_query(
            session,
            user,
            primary_query,
            llm_client,
            ModelConfig.from_settings(settings),
            QueryOptions(
                limit=fetch_limit,
                min_score=min_score,
                candidate_limit=candidate_limit or 40,
                collection_name=single,
                intent=intent,
                skip_rerank=not rerank,
                preexpanded=preexpanded,
            ),
        )

    results = _filter_by_collections(results, effective_collections)
    results = results[:limit]

    items = []
    for r in results:
        snippet_info = extract_snippet(r.body, primary_query, 300, r.best_chunk_pos, None, intent)
        items.append(
            {
                "docid": f"#{r.docid}",
                "file": r.display_path,
                "title": r.title,
                "score": _round2(r.score),
                "context": r.context,
                "line": snippet_info.line,
                "snippet": add_line_numbers(snippet_info.snippet, snippet_info.line),
            }
        )
    return items, primary_query


def _format_search_summary(results: list[dict[str, Any]], query: str) -> str:
    if not results:
        return f'No results found for "{query}"'
    plural = "" if len(results) == 1 else "s"
    lines = [f'Found {len(results)} result{plural} for "{query}":\n']
    for r in results:
        lines.append(f"{r['docid']} {round(r['score'] * 100)}% {r['file']} - {r['title']}")
    return "\n".join(lines)


_QUERY_TOOL_DESCRIPTION = """Search the knowledge base using a query document — one or more \
typed sub-queries combined for best recall.

Each result includes a `line` field with the absolute 1-indexed line of the best match in \
the source markdown. To read more context around a hit, call \
`get(file, fromLine = max(1, line - 20), maxLines = 80, lineNumbers = true)`.

## Query Types

**lex** — BM25 keyword search. Fast, exact, no LLM needed.
Full lex syntax:
- `term` — prefix match ("perf" matches "performance")
- `"exact phrase"` — phrase must appear verbatim
- `-term` or `-"phrase"` — exclude documents containing this

**vec** — Semantic vector search. Write a natural language question. Finds documents by \
meaning, not exact words.

**hyde** — Hypothetical document. Write 50-100 words that look like the answer. Often the \
most powerful for nuanced topics.

## Strategy

Combine types for best results. First sub-query gets 2x weight — put your strongest signal first.

| Goal | Approach |
|------|----------|
| General search (recommended) | Pass `query` — auto-expanded into typed variants, fused, reranked |
| Know exact term/name | `lex` only |
| Concept search | `vec` only |
| Best recall | `lex` + `vec` |
| Complex/nuanced | `lex` + `vec` + `hyde` |
| Unknown vocabulary | Pass `query` with natural language so the server auto-expands it |
"""


def _register_query_tool(mcp: FastMCP, llm_client: LlmClient) -> None:
    @mcp.tool(
        name="query",
        title="Query",
        description=_QUERY_TOOL_DESCRIPTION,
        annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False),
    )
    async def query(
        query: Annotated[
            str | None,
            Field(
                default=None,
                description="Plain-text query, auto-expanded by the SDK into lex/vec/hyde "
                "variants, fused via RRF and reranked. Recommended default for most searches. "
                "Mutually exclusive with 'searches'.",
            ),
        ] = None,
        searches: Annotated[
            list[SubSearch] | None,
            Field(
                default=None,
                max_length=10,
                description="Typed sub-queries to execute (lex/vec/hyde). First gets 2x "
                "weight. Use for precise control over retrieval strategy. Mutually exclusive "
                "with 'query'.",
            ),
        ] = None,
        limit: Annotated[int, Field(default=10, description="Max results (default: 10)")] = 10,
        minScore: Annotated[
            float, Field(default=0.0, description="Min relevance 0-1 (default: 0)")
        ] = 0.0,
        candidateLimit: Annotated[
            int | None,
            Field(
                default=None,
                description="Maximum candidates to rerank (default: 40, lower = faster but "
                "may miss results)",
            ),
        ] = None,
        collections: Annotated[
            list[str] | None, Field(default=None, description="Filter to collections (OR match)")
        ] = None,
        intent: Annotated[
            str | None,
            Field(
                default=None,
                description="Background context to disambiguate the query. Example: "
                "query='performance', intent='web page load times and Core Web Vitals'. "
                "Does not search on its own.",
            ),
        ] = None,
        rerank: Annotated[
            bool,
            Field(
                default=True,
                description="Rerank results using LLM (default: true). Set to false for "
                "faster results on CPU-only machines.",
            ),
        ] = True,
    ) -> CallToolResult:
        if not query and not searches:
            logger.warning("query tool rejected: neither 'query' nor 'searches' given")
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text="Error: provide either 'query' (plain text) or 'searches' "
                        "(typed sub-queries)",
                    )
                ],
                isError=True,
            )
        if query and searches:
            logger.warning(
                "query tool rejected: 'query' and %d 'searches' entries are mutually exclusive",
                len(searches),
            )
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text="Error: 'query' and 'searches' are mutually exclusive; "
                        "provide only one",
                    )
                ],
                isError=True,
            )
        if searches:
            invalid = validate_typed_queries(
                [ExpandedQuery(s.type, s.query) for s in searches]
            )
            if invalid is not None:
                logger.warning("query tool rejected: %s", invalid)
                return CallToolResult(
                    content=[TextContent(type="text", text=f"Error: {invalid}")],
                    isError=True,
                )

        items, primary_query = await _run_query_search(
            llm_client=llm_client,
            query=query,
            searches=searches,
            limit=limit,
            min_score=minScore,
            candidate_limit=candidateLimit,
            collections=collections,
            intent=intent,
            rerank=rerank,
        )
        return CallToolResult(
            content=[TextContent(type="text", text=_format_search_summary(items, primary_query))],
            structuredContent={"results": items},
        )


# =============================================================================
# Tool: get
# =============================================================================


def _parse_get_lookup(
    file: str, from_line: int | None, max_lines: int | None
) -> tuple[str, int | None, int | None]:
    import re

    range_match = re.search(r":(\d+):(\d+)$", file)
    if range_match:
        if from_line is None:
            from_line = int(range_match.group(1))
        if max_lines is None:
            max_lines = int(range_match.group(2))
        lookup = file[: range_match.start()]
    else:
        line_match = re.search(r":(\d+)$", file)
        if line_match and from_line is None:
            from_line = int(line_match.group(1))
            lookup = file[: line_match.start()]
        else:
            lookup = file
    if from_line is not None:
        from_line = max(1, from_line)
    return lookup, from_line, max_lines


def _register_get_tool(mcp: FastMCP) -> None:
    @mcp.tool(
        name="get",
        title="Get Document",
        description="Retrieve the full content of a document by its file path or docid. "
        "Use paths or docids (#abc123) from search results. Suggests similar files if not "
        "found.",
        annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False),
    )
    async def get(
        file: Annotated[
            str,
            Field(
                description="File path or docid from search results. Supports a line-range "
                "suffix: 'pages/meeting.md:100' starts at line 100; 'pages/meeting.md:100:40' "
                "(or '#abc123:100:40') reads 40 lines from line 100."
            ),
        ],
        fromLine: Annotated[
            int | None, Field(default=None, description="Start from this line number (1-indexed)")
        ] = None,
        maxLines: Annotated[
            int | None, Field(default=None, description="Maximum number of lines to return")
        ] = None,
        lineNumbers: Annotated[
            bool,
            Field(
                default=True,
                description="Add line numbers to output (format: 'N: content'). On by "
                "default; set false for raw content.",
            ),
        ] = True,
    ) -> CallToolResult:
        lookup, parsed_from, parsed_max = _parse_get_lookup(file, fromLine, maxLines)

        async with get_session() as session:
            user = await get_current_user(session)
            await session.commit()
            result = await find_document(session, user, lookup)

        if isinstance(result, DocumentNotFound):
            msg = f"Document not found: {file}"
            if result.similar_files:
                suggestions = "\n".join(f"  - {s}" for s in result.similar_files)
                msg += f"\n\nDid you mean one of these?\n{suggestions}"
            return CallToolResult(content=[TextContent(type="text", text=msg)], isError=True)

        assert isinstance(result, DocumentDetail)
        body = result.body
        start_line = parsed_from or 1
        if parsed_from is not None or parsed_max is not None:
            lines = body.split("\n")
            start = start_line - 1
            end = start + parsed_max if parsed_max is not None else len(lines)
            body = "\n".join(lines[start:end])

        text = body
        if lineNumbers:
            text = add_line_numbers(text, start_line)
        if result.context:
            text = f"<!-- Context: {result.context} -->\n\n{text}"

        return CallToolResult(
            content=[
                EmbeddedResource(
                    type="resource",
                    resource=TextResourceContents(
                        uri=f"marq://{_encode_qmd_path(result.display_path)}",
                        mimeType="text/markdown",
                        text=text,
                    ),
                )
            ]
        )


# =============================================================================
# Tool: multi_get
# =============================================================================


def _register_multi_get_tool(mcp: FastMCP) -> None:
    @mcp.tool(
        name="multi_get",
        title="Multi-Get Documents",
        description="Retrieve multiple documents by glob pattern (e.g., "
        "'journals/2025-05*.md') or comma-separated list. Skips files larger than maxBytes.",
        annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False),
    )
    async def multi_get_tool(
        pattern: Annotated[
            str, Field(description="Glob pattern or comma-separated list of file paths")
        ],
        maxLines: Annotated[
            int | None, Field(default=None, description="Maximum lines per file")
        ] = None,
        maxBytes: Annotated[
            int,
            Field(
                default=DEFAULT_MULTI_GET_MAX_BYTES,
                description="Skip files larger than this (default: 65536 = 64KB)",
            ),
        ] = DEFAULT_MULTI_GET_MAX_BYTES,
        lineNumbers: Annotated[
            bool,
            Field(
                default=True,
                description="Add line numbers to output (format: 'N: content'). On by "
                "default; set false for raw content.",
            ),
        ] = True,
    ) -> CallToolResult:
        async with get_session() as session:
            user = await get_current_user(session)
            await session.commit()
            results = await multi_get(
                session, user, pattern, max_lines=maxLines, max_bytes=maxBytes,
                line_numbers=lineNumbers,
            )

        if not results:
            return CallToolResult(
                content=[TextContent(type="text", text=f"No files matched pattern: {pattern}")],
                isError=True,
            )

        ContentBlock = TextContent | ImageContent | AudioContent | ResourceLink | EmbeddedResource
        content: list[ContentBlock] = []
        for r in results:
            if r.skipped:
                content.append(
                    TextContent(
                        type="text",
                        text=f"[SKIPPED: {r.display_path} - {r.skip_reason}. Use 'get' with "
                        f'file="{r.display_path}" to retrieve.]',
                    )
                )
                continue
            content.append(
                EmbeddedResource(
                    type="resource",
                    resource=TextResourceContents(
                        uri=f"marq://{_encode_qmd_path(r.display_path)}",
                        mimeType="text/markdown",
                        text=r.body,
                    ),
                )
            )
        return CallToolResult(content=content)


# =============================================================================
# Tool: status
# =============================================================================


def _register_status_tool(mcp: FastMCP, embed_model: str) -> None:
    @mcp.tool(
        name="status",
        title="Index Status",
        description="Show the status of the QMD index: collections, document counts, and "
        "health information.",
        annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False),
    )
    async def status() -> CallToolResult:
        async with get_session() as session:
            user = await get_current_user(session)
            await session.commit()
            info = await get_status(session, user)
            health = await get_vector_index_health(session, user, embed_model)

        summary = [
            "marq Index Status:",
            f"  Total documents: {info.total_documents}",
            f"  Needs embedding: {health.needs_embedding}",
            f"  Vector index: {'yes' if health.has_vector_index else 'no'}",
            f"  Collections: {len(info.collections)}",
        ]
        for c in info.collections:
            summary.append(f"    - {c.name}: {c.path} ({c.doc_count} docs)")

        structured = {
            "totalDocuments": info.total_documents,
            "needsEmbedding": health.needs_embedding,
            "hasVectorIndex": health.has_vector_index,
            "collections": [
                {
                    "name": c.name,
                    "path": c.path,
                    "pattern": c.pattern,
                    "documents": c.doc_count,
                    "lastUpdated": c.last_updated.isoformat() if c.last_updated else None,
                }
                for c in info.collections
            ],
        }
        return CallToolResult(
            content=[TextContent(type="text", text="\n".join(summary))],
            structuredContent=structured,
        )


# =============================================================================
# Resource: marq://{path} - see module docstring for why this bypasses
# FastMCP's own @mcp.resource() decorator.
# =============================================================================


def _register_document_resource(mcp: FastMCP) -> None:
    @mcp._mcp_server.read_resource()  # type: ignore[no-untyped-call,untyped-decorator]
    async def _read_resource(uri: Any) -> list[ReadResourceContents]:
        raw = str(uri)
        path = raw[len("marq://") :] if raw.startswith("marq://") else raw
        decoded_path = unquote(path)

        async with get_session() as session:
            user = await get_current_user(session)
            await session.commit()
            result = await find_document(session, user, decoded_path)

        if isinstance(result, DocumentNotFound):
            return [
                ReadResourceContents(
                    content=f"Document not found: {decoded_path}", mime_type="text/markdown"
                )
            ]

        assert isinstance(result, DocumentDetail)
        text = add_line_numbers(result.body)
        if result.context:
            text = f"<!-- Context: {result.context} -->\n\n{text}"
        return [ReadResourceContents(content=text, mime_type="text/markdown")]


# =============================================================================
# REST endpoints (HTTP transport only) - GET /health, POST /query (+
# /search alias) offer the same search as the `query` tool without
# requiring the MCP JSON-RPC protocol. `/mcp` itself (full JSON-RPC,
# session management via the `mcp-session-id` header) needs no manual
# wiring here - it's already part of `mcp.streamable_http_app()`, handled
# internally by the SDK's own StreamableHTTPSessionManager.
# =============================================================================


def _is_int(value: Any) -> bool:
    """`isinstance(x, int)` alone accepts JSON true/false (Python's bool
    subclasses int), so `"limit": true` would quietly run with limit 1 -
    this is the standard fence."""
    return isinstance(value, int) and not isinstance(value, bool)


def _register_rest_routes(mcp: FastMCP, start_time: float, llm_client: LlmClient) -> None:
    from starlette.requests import Request
    from starlette.responses import JSONResponse

    @mcp.custom_route("/health", methods=["GET"])  # type: ignore[untyped-decorator]
    async def health(request: Request) -> JSONResponse:
        return JSONResponse({"status": "ok", "uptime": int(time.time() - start_time)})

    async def query_rest(request: Request) -> JSONResponse:
        try:
            params = await request.json()
        except ValueError:
            logger.warning("REST /query rejected: body is not valid JSON")
            return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

        raw_searches = params.get("searches")
        if not isinstance(raw_searches, list):
            # Payload *shape* only - the query text is user data (see log.py).
            logger.warning(
                "REST /query rejected: 'searches' is %s, expected array",
                type(raw_searches).__name__,
            )
            return JSONResponse(
                {"error": "Missing required field: searches (array)"}, status_code=400
            )
        try:
            searches = [
                SubSearch(type=s.get("type"), query=str(s.get("query") or ""))
                for s in raw_searches
            ]
        except (ValidationError, AttributeError, TypeError) as exc:
            logger.warning(
                "REST /query rejected: malformed 'searches' entry among %d (%s)",
                len(raw_searches),
                type(exc).__name__,
            )
            return JSONResponse(
                {"error": "Each searches entry must be an object with a valid type (lex/vec/hyde)"},
                status_code=400,
            )
        invalid = validate_typed_queries([ExpandedQuery(s.type, s.query) for s in searches])
        if invalid is not None:
            logger.warning("REST /query rejected: %s", invalid)
            return JSONResponse({"error": invalid}, status_code=400)
        raw_collections = params.get("collections")
        collections = (
            [str(c) for c in raw_collections] if isinstance(raw_collections, list) else None
        )
        intent = params.get("intent") if isinstance(params.get("intent"), str) else None

        items, _primary_query = await _run_query_search(
            llm_client=llm_client,
            query=None,
            searches=searches,
            limit=params["limit"] if _is_int(params.get("limit")) else 10,
            min_score=(
                params["minScore"]
                if _is_int(params.get("minScore")) or isinstance(params.get("minScore"), float)
                else 0.0
            ),
            candidate_limit=(
                params["candidateLimit"] if _is_int(params.get("candidateLimit")) else None
            ),
            collections=collections,
            intent=intent,
            rerank=params["rerank"] if isinstance(params.get("rerank"), bool) else True,
        )
        # The REST alias uses the full marq:// URI for "file" - unlike the
        # `query` tool's bare display_path - matching the TS reference's
        # own (undocumented) discrepancy between its MCP tool and REST
        # endpoint output shapes.
        for item in items:
            item["file"] = f"marq://{_encode_qmd_path(str(item['file']))}"
        return JSONResponse({"results": items})

    mcp.custom_route("/query", methods=["POST"])(query_rest)
    mcp.custom_route("/search", methods=["POST"])(query_rest)


# =============================================================================
# Server construction - shared by stdio and HTTP transports.
# =============================================================================


async def create_mcp_server(
    *, http: bool = False, host: str = "127.0.0.1", port: int = 8181
) -> FastMCP:
    """Build the MCP server with its tools, resource and instructions.

    Async because the instructions are built from live index state, which
    means a database round trip before the server can be constructed.

    Args:
        http: Also register the REST routes (`/health`, `/query`,
            `/search`). They only exist under the HTTP transport; `/mcp`
            itself is handled by the SDK.
        host: Bind address, HTTP transport only.
        port: Bind port, HTTP transport only.

    Returns:
        A configured `FastMCP`, ready for either transport.
    """
    settings = get_settings()
    # Never stdout: the stdio transport *is* JSON-RPC over stdout, so a
    # stray log line corrupts the protocol. setup_logging only ever
    # writes to stderr or a file, and is idempotent, so this is safe
    # whether or not the CLI entry point already configured it.
    setup_logging(settings.log_level, settings.log_file)

    async with get_session() as session:
        user = await get_current_user(session)
        await session.commit()
        instructions = await build_instructions(session, user, settings.embed_model)

    # One connection pool for the server's whole lifetime - every query
    # reuses keep-alive connections instead of paying TCP/TLS setup per
    # call (the CLI can't do better, one process per command; a
    # long-lived server can). Closed via the lifespan hook.
    llm_client = LlmClient(settings.llm_base_url)

    @asynccontextmanager
    async def _lifespan(_server: FastMCP) -> AsyncIterator[None]:
        try:
            yield
        finally:
            await llm_client.aclose()

    mcp = FastMCP(
        name="marq",
        instructions=instructions,
        host=host,
        port=port,
        json_response=True,
        lifespan=_lifespan,
    )
    # FastMCP's own constructor has no `version` parameter - it always
    # constructs its internal low-level Server with version=None, which
    # falls back to reporting the installed `mcp` SDK's own version in
    # initialize()'s serverInfo, not qmd-py's. Set it directly (a plain
    # public attribute on the low-level Server) so a client actually sees
    # which qmd-py release it's talking to.
    mcp._mcp_server.version = version("qmd-py")
    _register_query_tool(mcp, llm_client)
    _register_get_tool(mcp)
    _register_multi_get_tool(mcp)
    _register_status_tool(mcp, settings.embed_model)
    _register_document_resource(mcp)
    if http:
        _register_rest_routes(mcp, time.time(), llm_client)
    return mcp
