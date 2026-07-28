# MCP server

marq exposes its search over the
[Model Context Protocol](https://modelcontextprotocol.io/) via the
official Python SDK's `FastMCP` — four tools plus a document resource,
over stdio or Streamable HTTP.

## Transports

```console
$ marq mcp                          # stdio - for editors/agents that spawn a subprocess
$ marq mcp --http --port 8181       # Streamable HTTP - for remote/persistent access
$ marq mcp --http --daemon          # HTTP, backgrounded
$ marq mcp stop                     # stop the background daemon
```

```console
$ marq mcp --http --daemon --port 8891
MCP daemon started (pid 1068308), listening on http://127.0.0.1:8891/mcp
Logs: /home/andrea/.cache/marq/marq.log (level INFO)

$ curl -s http://127.0.0.1:8891/health
{"status":"ok","uptime":0}

$ marq mcp stop
Stopped MCP daemon (pid 1068308).
```

`--daemon` writes its PID and logs under `~/.cache/marq/`:

- `mcp.pid` — read by `mcp stop`, which sends `SIGTERM` and cleans it
  up, including a stale PID file left behind if the process had already
  died.
- `marq.log` — the daemon's own log, size-rotated (5 MB, 3 backups) and
  at `INFO` by default, so a server running for weeks can't fill the
  disk. `MARQ_LOG_LEVEL`/`MARQ_LOG_FILE` override both if set; see
  [Configuration](configuration.md#marq_log_level).
- `mcp-stdio.log` — anything the process writes outside the logging
  system (a startup traceback, uvicorn's own banner). Truncated at each
  start.

At `INFO` the log carries one line per request, correlated by a short
id so concurrent tool calls stay followable:

```text
2026-07-28 16:56:34,613 WARNING qmd_py.mcp.server [rest-3f3ad5] REST /query rejected: 'searches' is str, expected array
2026-07-28 16:56:34,614 INFO    qmd_py.mcp.server [rest-3f3ad5] POST /query: status=400 2ms
2026-07-28 16:56:34,749 INFO    qmd_py.search.hybrid [rest-99ce16] query: subqueries=3 candidates=40 reranked=yes results=9 812ms
```

Note what is *not* there: no query text and no document content. Those
are logged only at `DEBUG` — see the privacy note in
[Configuration](configuration.md#marq_log_file).

If you bind a non-loopback `--host`, the server warns at startup: the
HTTP transport has no authentication yet, so anyone who can reach the
port can read every indexed document.

## Tools

| Tool | Purpose |
|---|---|
| `query` | Hybrid search — same pipeline as the CLI's `query` command |
| `get` | Retrieve one document by path or docid |
| `multi_get` | Retrieve multiple documents by glob or comma-separated list |
| `status` | Index status: collections, document counts, embedding health |

`query` accepts **either** a plain-text `query` string (auto-expanded,
same as `marq query "..."`) **or** an explicit `searches` array of typed
`{type: "lex"|"vec"|"hyde", query: "..."}` sub-queries (same as marq's
structured `lex:`/`vec:`/`hyde:` query documents) — never both; passing
neither or both is a tool error, not a silent fallback. See
[Search & query](commands/search-and-query.md) for what each sub-query
type means.

Tool input schema field names deliberately stay camelCase (`minScore`,
`candidateLimit`, `fromLine`, `lineNumbers`, ...) rather than idiomatic
Python `snake_case` — this is a wire-compatibility surface for MCP
clients, kept consistent with the equivalent fields across marq's
tooling, not internal Python code.

## Resource

`marq://collection/path` — read a document as an MCP resource. No
`list()`: documents are discovered via the search tools, not enumerated.
A not-found path returns ordinary text content (`"Document not found:
..."`), not an MCP protocol error, matching how `get` degrades on the
CLI side.

The resource handler is registered directly against the low-level
`mcp._mcp_server`, bypassing FastMCP's own `@mcp.resource()` decorator —
that decorator's template matching only supports a single path segment
per `{param}` (`[^/]+`), so it can't express a slash-spanning
`marq://collection/nested/path.md`. The low-level `read_resource()`
handler receives the full raw URI and parses it directly instead.

## Dynamic instructions

The `initialize` response's `instructions` field is built from live
index state — total document count, collection names, and whether
vector embeddings are missing/stale — so an agent gets useful context
immediately, without needing a `status` tool call first just to learn
what's searchable.

## HTTP transport: REST endpoints alongside JSON-RPC

When running over `--http`, three plain REST endpoints exist alongside
full MCP JSON-RPC at `/mcp`:

- `GET /health` — `{"status": "ok", "uptime": <seconds>}`.
- `POST /query` (alias `POST /search`) — the same hybrid search as the
  `query` tool, callable without speaking MCP JSON-RPC at all. Body:
  `{"searches": [{"type": "lex", "query": "..."}], "collections": [...],
  "limit": 10, "minScore": 0, "intent": "...", "rerank": true}`.
  Returns `{"results": [...]}` with each result's `file` as a full
  `marq://collection/path` URI (unlike the `query` *tool*, whose `file`
  field is the bare collection-relative path — a small, deliberate
  difference between the two output shapes).

Full JSON-RPC session management (the `mcp-session-id` header,
`initialize`/`notifications/initialized` handshake, session lookup) is
handled entirely by the MCP SDK's own `StreamableHTTPSessionManager` —
serving `mcp.streamable_http_app()` directly (rather than mounting it
into a separate hand-built FastAPI app) means its lifespan is already
wired correctly, so there's no manual session-map bookkeeping to get
wrong.
