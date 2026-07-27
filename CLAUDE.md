# marq

Centralized markdown/code search over Postgres/pgvector, with a hybrid
BM25 + vector + LLM-reranking search pipeline over plain HTTP, a CLI, and
an MCP server for AI agents. Inspired by [qmd](https://github.com/tobi/qmd)
but a from-scratch Python rewrite, not a fork — different storage
(shared Postgres/pgvector instead of per-project SQLite), different LLM
story (pure HTTP client, no local model loading), and a schema built for
eventual multi-user access control. See README.md for the full feature
list and command reference.

Use `uv run marq <command>` (or activate `.venv` and run `marq` directly)
throughout — this is a `uv`-managed project, not `pip`/`poetry`.

## Commands

```sh
marq collection add . --name <name>       # Create/index a collection
marq collection list                      # List all collections with details
marq collection remove <name>             # Remove a collection by name
marq collection rename <old> <new>        # Rename a collection
marq ls [collection[/path]]               # List collections or files in a collection
marq context add [path] "text"            # Add context for path (defaults to current dir)
marq context list                         # List all contexts
marq context check                        # Check for collections/paths missing context
marq context rm <path>                    # Remove context
marq get <file>[:from[:count]]            # Get by path or docid (#abc123); optional line range
marq multi-get <pattern>                  # Get multiple docs by glob or comma-separated list
marq status                               # Show index status and collections
marq doctor                               # Diagnose Postgres/pgvector/migration/LLM-router health
marq update                               # Re-index collections; configured update commands run first
marq embed [-c <name>] [--model <slug>]   # Generate/refresh vector embeddings
marq query <query>                        # Hybrid search: expansion + RRF fusion + reranking (recommended)
marq search <query>                       # Full-text keyword search (BM25, no LLM)
marq vsearch <query>                      # Vector similarity search (no reranking)
marq bench <fixture.json>                 # Run search-quality benchmarks
marq mcp                                  # Start MCP server (stdio transport)
marq mcp --http [--port N]                # Start MCP server (Streamable HTTP)
marq mcp --http --daemon                  # Start as background daemon
marq mcp stop                             # Stop background MCP daemon
marq skill show / install                 # Show or install the bundled agent skill
marq skills list / get / path             # Discover bundled skills generically
marq cleanup                              # Remove orphaned content/embeddings/documents
```

## Query syntax (`marq query`)

A query is either a single auto-expanded string, or a multi-line document
where every line is typed with `lex:`, `vec:`, `hyde:`, or an optional
`intent:`:

```sh
marq query "how does auth work"                                      # single-line -> implicit expand
marq query "$(printf 'lex: CAP theorem\nvec: consistency tradeoffs')" # typed query document
marq query "$(printf 'lex: \"exact phrase\" sports -baseball')"       # phrase + negation lex search
```

The first sub-query (whether the implicit expansion or the first typed
line) gets 2x RRF weight — put the strongest signal first. `--explain`
(with the default `cli` format or `--format json`) shows the RRF
rank/weight and rerank score behind each result. `--no-rerank` skips the
LLM reranking pass for faster, lower-quality results.

## Development

```sh
uv sync --all-extras                 # install dependencies, incl. pytest/ruff/mypy/docs
uv run marq <command>                # run from source
uv run pytest                        # run the test suite
uv run pytest -m "not integration"   # skip tests needing real Postgres/LLM
uv run ruff check --fix .
uv run mypy src alembic tests
uv run alembic check                 # verify models match the latest migration
uv run alembic upgrade head          # apply migrations
uv run zensical build --strict       # build the docs site (docs/, zensical.toml)
uv run zensical serve                # preview it locally at localhost:8000
```

Docs site: `zensical` (TOML-config static site generator) +
`mkdocstrings`/`mkdocs-click` for auto-generated API/CLI reference —
under the `docs` optional-dependencies extra. No CI/deployment wiring
yet (Forgejo CI is deliberately skipped for now; GitHub Actions/Pages is
the plan once the project moves there) — build and preview locally.

Local dev Postgres: `cp .env.dev .env && podman-compose up -d` starts a
disposable `pgvector/pgvector:pg16` container on `localhost:5433`,
version-matched to the production instance. An optional local LLM router
is available too (`llm-stack/README.md`) — entirely for convenience, not
required; `MARQ_LLM_BASE_URL` can point anywhere OpenAI-endpoint-shaped.

### Testing conventions

- Tests use a **fresh, isolated Postgres schema per test run**
  (`qmd_test_<uuid>`, see `tests/conftest.py`), created against the same
  instance `MARQ_POSTGRES_URL` points at and dropped at teardown — not a
  separate throwaway database, since that would need its own
  `vector`/`pg_trgm` extension install.
- Tables in that isolated schema are created directly from
  `SQLModel.metadata`, **not** via Alembic — these tests care about
  behavior, not migration history.
- Integration tests (`@pytest.mark.integration`) hit real Postgres and
  the real LLM router; run `pytest -m "not integration"` to skip them
  when neither is reachable.
- `tests/fixtures/sample-collection/` is a small, frozen, checked-in
  multi-language project (`.md`/`.py`/`.js`/`.ts`) — a manageable
  stand-in for a real external collection, usable without SSH access or
  external data. `tests/fixtures/bench-sample-collection.json` is a
  matching benchmark fixture for `marq bench`.
- **CLI and MCP commands are pytested end-to-end** via two conftest
  fixtures that redirect `db/engine.py`'s process-global (`lru_cache`'d)
  `get_engine()`/`get_session()` at a throwaway schema: `marq` (sync,
  click `CliRunner` — see `tests/test_cli.py`, `test_cli_query.py`) and
  `mcp_env` (async, a real MCP `ClientSession` over the SDK's in-memory
  transport — see `tests/test_mcp_server.py`). Both patch
  `MARQ_POSTGRES_SCHEMA` and call `db/engine.py`'s `reset_engine()`.
  CLI tests must be **sync**: every command body ends in
  `cli/runtime.py`'s `asyncio.run()`, which refuses to start inside a
  running loop. `reset_engine()` also has to run between invocations in
  one process, since pooled connections don't survive their loop.
- `tests/test_cli_skills.py` needs neither Postgres nor the router, so
  it runs in the default `-m "not integration"` suite.
- **Never skip a live-verification pass on a claim you can't prove from
  pytest alone.** Several real bugs in this project (a Postgres
  `ts_rank` weight-range error, a token-budget overflow against the
  reranker, an ACL-adjacent FK-cascade gap) were only caught by actually
  running the CLI/MCP server against real infrastructure, not by unit
  tests.

## Architecture

- **Storage**: Postgres + pgvector, one schema (`MARQ_POSTGRES_SCHEMA`,
  default `qmd_py`), managed by Alembic. Content is content-addressed by
  SHA-256 hash in one table shared across every collection — indexing an
  unchanged file (a second git worktree, a merged branch) costs nothing
  extra. Each embedding model gets its own physical table
  (`embeddings_<slug>`), created idempotently on first use; switching
  models is additive, never destructive.
- **Search**: `tsvector`/`ts_rank` (BM25), pgvector HNSW (vector
  similarity), application-level Reciprocal Rank Fusion to combine
  ranked lists, and an LLM reranking pass over the fused candidates.
- **LLM client**: pure `httpx` HTTP client against an OpenAI-style
  endpoint (`src/qmd_py/llm/client.py`) — no local model loading, no
  GPU/device concept in the Python process itself.
- **ACL**: schema and call sites (`can_access()`, called at every
  read/write choke point — see `src/qmd_py/auth.py`) are structured for
  real multi-user access control, but the check itself is mocked to
  always allow; the real use case today is a single local user. Swapping
  in a real check is meant to be additive — one function body, not a
  rearchitecture — and `tests/test_acl_gating.py` proves every choke
  point actually gates once a real check lands.
- **CLI**: `click`; one command (or command group) per file under
  `src/qmd_py/cli/commands/`. `main.py`'s `main()` wraps the whole CLI to
  turn a missing/invalid `MARQ_*` setting into a short message instead of
  a raw pydantic traceback.
- **MCP server**: the official `mcp` Python SDK's `FastMCP`
  (`src/qmd_py/mcp/server.py`). The `marq://{path}` document resource is
  registered against the low-level `mcp._mcp_server` directly (bypassing
  FastMCP's own `@mcp.resource()` decorator, whose template matching
  can't express a slash-spanning path segment).
- **Python package name stays `qmd_py`** (distribution `qmd-py`) even
  though the CLI/MCP product surface is branded `marq` — the package
  name is an internal implementation detail no CLI/MCP user ever sees;
  renaming it is a separate, larger, currently out-of-scope change.

## Important: verify live, and don't run mutating commands unasked

- Never run `marq collection add`, `marq embed`, `marq update`, or
  `marq cleanup` against real data without being asked — these mutate
  the index.
- Never modify the Postgres database directly (raw `psql`/SQL against
  marq's tables) — go through the CLI or the service layer
  (`src/qmd_py/store.py`) so ACL/content-addressing invariants hold.
- When verifying a change against real infrastructure, clean up
  afterward: remove any collection you added for the check
  (`marq collection remove <name>`) on every environment you touched.
