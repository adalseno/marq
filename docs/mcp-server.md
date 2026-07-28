# MCP server

marq exposes its search over the
[Model Context Protocol](https://modelcontextprotocol.io/) via the
official Python SDK's `MCPServer` — four tools plus a document resource,
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
  system (a startup traceback, uvicorn's own banner). Rotated at each
  start: the previous run moves to `mcp-stdio.log.1`, so a crashed
  daemon's traceback survives the restart you do to investigate it.

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
port can read every indexed document. See
[Deployment: TLS and authentication](#deployment-tls-and-authentication)
for what to put in front of it.

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
server, bypassing `MCPServer`'s own `@mcp.resource()` decorator — that
decorator's template matching only supports a single path segment per
`{param}` (`[^/]+`), so it can't express a slash-spanning
`marq://collection/nested/path.md`. The low-level handler receives the
full raw URI and parses it directly instead, via the SDK 2.x
`add_request_handler("resources/read", ...)` registry.

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

## Deployment: TLS and authentication

marq serves plain HTTP and has **no built-in TLS** — there is no
`--ssl`/certificate option, and the MCP SDK's
`run_streamable_http_async()` takes no TLS parameters. (The SDK's
`TransportSecuritySettings` sounds relevant but is DNS-rebinding
protection — `Host`/`Origin` validation — not encryption.) Put a reverse
proxy in front.

### Why the proxy also has to authenticate

This is the part to not skip, and it follows directly from marq's
[ACL status](architecture.md): `can_access()` is called at every
read/write choke point, but **its body is mocked to allow everything**.
There are therefore two separate gaps, not one:

- **No authentication** — the HTTP transport has no notion of *who* is
  calling. No bearer token, no client certificate, nothing to identify a
  caller.
- **No authorization** — `can_access()` returns "allowed" regardless, so
  even a known caller would not be restricted.

The second is the one people notice; the first is why fixing only the
second would not help. A real `can_access()` needs an authenticated
identity to make a decision *about*, which is why the project plans them
together — the startup warning's own wording is "until real ACL (and with
it a bearer-token check) lands".

Until then, **TLS alone buys you an encrypted channel to an
unauthenticated index**: it stops eavesdropping, not access. So the proxy
does both jobs. Once real ACL and a token check land, the proxy's auth
layer becomes belt-and-braces rather than the only thing standing there,
and can be relaxed to taste.

None of this applies to a single-user machine with marq on loopback,
which is the default and the shape it is designed for. The warning fires
precisely when you leave that shape.

### Caddy example

Keep marq on loopback — the default — and let the proxy be the only
thing listening publicly:

```bash
marq mcp --http --daemon          # 127.0.0.1:8181
```

```caddyfile
marq.example.com {
    basic_auth {
        alice $2a$14$...          # caddy hash-password --plaintext 'secret'
    }

    reverse_proxy host.containers.internal:8181 {
        flush_interval -1          # do not buffer: GET /mcp can stream SSE
    }
}
```

```yaml
services:
  caddy:
    image: docker.io/library/caddy:2-alpine     # ~50 MB
    ports: ["80:80", "443:443"]
    volumes:
      - ./Caddyfile:/etc/caddy/Caddyfile:ro
      - caddy_data:/data                        # certificates — must persist
      - caddy_config:/config
    extra_hosts:
      - "host.containers.internal:host-gateway"

volumes:
  caddy_data:
  caddy_config:
```

Details worth knowing:

- `basic_auth` is the Caddy ≥ 2.8 spelling; earlier versions use
  `basicauth`. For something richer, `forward_auth` fronts an OAuth2
  proxy and `client_auth` does mTLS.
- `flush_interval -1` disables response buffering. `POST /mcp` returns
  plain JSON (marq runs the transport with `json_response=True`), but the
  Streamable HTTP spec still allows `GET /mcp` to hold an SSE stream, and
  a buffering proxy stalls it.
- Caddy preserves the client's `Host` header by default, unlike nginx —
  which matters if you ever enable the SDK's DNS-rebinding protection.
- marq is **not containerised** (there is no Dockerfile), so it runs on
  the host while Caddy runs in a container; hence the host-gateway hop.
  Podman 4.7+ resolves `host.containers.internal` by itself, Docker on
  Linux needs the `extra_hosts` line. Caddy is also a single static
  binary — running it directly on the host and proxying to
  `127.0.0.1:8181` sidesteps the whole question.
- **Certificates**: a public DNS name needs nothing, Caddy provisions and
  renews via Let's Encrypt automatically — just persist `caddy_data` or
  you re-issue on every restart and will hit rate limits. An internal
  hostname (`marq.lan`) can't be validated by ACME: add `tls internal` and
  Caddy issues from its own local CA, whose root you then install on the
  clients.

Outbound TLS already works, incidentally: `MARQ_LLM_BASE_URL` is passed
straight to `httpx`, so an `https://` router endpoint needs no
configuration.
