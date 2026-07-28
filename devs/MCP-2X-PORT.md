# Porting to the `mcp` SDK 2.x

What changed between `mcp` 1.28.1 and 2.0.0, and how marq was moved
across. Written during the port so the next SDK bump has a map.

The whole delta lands in three files — `src/qmd_py/mcp/server.py`,
`src/qmd_py/cli/commands/mcp.py`, `tests/test_mcp_server.py` — plus the
dependency constraint. Nothing outside the MCP surface was touched.

## Why the ceiling existed first

`pyproject.toml` declared `mcp>=1.2` with no upper bound, so a fresh
resolve picked up 2.0.0 while `uv.lock` pinned 1.28.1. The test suite
passed against the lock; a clean install of the built wheel got 2.0 and
died at import with `No module named 'mcp.server.fastmcp'`. That is what
the `mcp>=1.8,<2` pin on `master` bought time to fix properly.

## The renames

| 1.x | 2.x |
|---|---|
| `mcp.server.fastmcp.FastMCP` | `mcp.server.mcpserver.MCPServer` |
| `server._mcp_server` | `server._lowlevel_server` |
| `mcp.types` camelCase fields (`isError`, `mimeType`, `structuredContent`, `serverInfo`, `readOnlyHint`) | snake_case (`is_error`, `mime_type`, `structured_content`, `server_info`, `read_only_hint`) |

The camelCase → snake_case change is **Python-side only**. The JSON wire
format still uses camelCase via pydantic aliases, verified live against
a running server:

```json
{"contents":[{"mimeType":"text/markdown","uri":"marq://docs/deep/nested/file.md"}]}
```

So MCP clients see no difference. The tool input schemas that
deliberately stay camelCase (`minScore`, `candidateLimit`, ...) are
unaffected — those are `Field(alias=...)` declarations of ours, not SDK
types.

## The three real changes

**1. `version` became a constructor parameter.** 1.x had no way to set
it, so the code poked `mcp._mcp_server.version = version("qmd-py")` after
construction; without that the SDK reported *its own* version in
`initialize()`'s `serverInfo`. 2.x takes `version=` directly. The hack
and its six-line comment are gone.

**2. The bind moved off the constructor.** `host`, `port` and
`json_response` are now arguments to `run_streamable_http_async()` and
`streamable_http_app()`. `create_mcp_server()` consequently takes no bind
address at all — it is transport-agnostic, and `cli/commands/mcp.py`'s
`_run_http` owns the address. The readiness log line lost its `bind=`
field for the same reason (the address isn't known at construction);
`_run_http` logs it when the transport actually starts.

**3. The low-level handler registry replaced the per-method decorators.**
This is the one that needed thought, because the `marq://{path}` resource
deliberately bypasses `@mcp.resource()` (whose template matching can't
express a slash-spanning path segment — see the module docstring).

```python
# 1.x
@mcp._mcp_server.read_resource()
async def _read_resource(uri: Any) -> list[ReadResourceContents]: ...

# 2.x
async def _read_resource(_ctx: Any, params: ReadResourceRequestParams) -> ReadResourceResult: ...
mcp._lowlevel_server.add_request_handler(
    "resources/read", ReadResourceRequestParams, _read_resource
)
```

Three differences beyond the spelling: handlers take `(ctx, params)`;
they return the result model directly rather than `list[...]` wrapped by
the SDK (`ServerResult` is a plain union alias in 2.x, not a callable
wrapper); and `params.uri` is a `str`, not an `AnyUrl`. The last one
suits this handler — the point of the bypass is to see the raw URI before
anything splits it on slashes.

Side benefit: `add_request_handler` is fully typed, so the
`# type: ignore[no-untyped-call,untyped-decorator]` the old decorator
needed is gone. `custom_route` is still untyped, so `src/` keeps exactly
one `type: ignore`.

## The test harness

`mcp.shared.memory.create_connected_server_and_client_session` was
**removed**; only `create_client_server_memory_streams` survives. That
helper was the foundation of every protocol-level test, so its plumbing —
run the server in a task group, hand the other stream ends to a
`ClientSession` — now lives explicitly in `test_mcp_server.py`'s
`_client()`. About fifteen lines. Nothing else in that file changed
except the field renames and dropping `AnyUrl` around resource URIs.

## What was verified

- Full suite: **424 passed** (integration included), `ruff` and
  `mypy src alembic tests` clean, `zensical build --strict` clean.
- The tool decorators were the main worry going in — `@mcp.tool()` builds
  JSON schemas by introspecting the signature, and `_with_request_id`
  sits underneath relying on `functools.wraps`. It works unchanged; all
  45 MCP tests passed on the first run of the ported harness.
- **Live**, because the run-call signature is the one path tests don't
  cover (they use `streamable_http_app()`, not
  `run_streamable_http_async()`): started `marq mcp --http --port 8899`
  and drove `GET /health`, the full `initialize` handshake (correct
  `serverInfo.version`, `content-type: application/json` confirming
  `json_response=True`, an `mcp-session-id` issued), `tools/list`
  (all four tools), and `resources/read` on a slash-spanning
  `marq://docs/deep/nested/file.md` — the bypass works over the wire.
  Then `marq mcp` over stdio: JSON-RPC on stdout, **stderr empty**, so
  the never-stdout rule and the `propagate = False` fix both still hold.

## Still open

Not attempted here, and not required by the port:

- 2.x adds `middleware`, `extensions`, `subscriptions`, `cache_hints` and
  a `request_state` mechanism to `MCPServer`. None are used. The
  `middleware` hook is the interesting one — it is the documented way to
  observe `initialize`, which is where a real ACL check would eventually
  want to sit.
- `MCPServer.custom_route` is still unannotated upstream; the one
  remaining `type: ignore` goes away if that is ever fixed.
