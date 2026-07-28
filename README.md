# marq

[![CI](https://github.com/adalseno/marq/actions/workflows/ci.yml/badge.svg)](https://github.com/adalseno/marq/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-adalseno.github.io%2Fmarq-blue)](https://adalseno.github.io/marq/)
[![Python](https://img.shields.io/badge/python-3.13%2B-blue)](https://www.python.org/)
[![Coverage](https://img.shields.io/badge/coverage-93%25-brightgreen)](devs/COVERAGE.md)

Centralized markdown/code search over Postgres/pgvector, with a hybrid
BM25 + vector + LLM-reranking search pipeline over plain HTTP, a CLI, and
an MCP server for AI agents.

**Documentation: [adalseno.github.io/marq](https://adalseno.github.io/marq/)**

marq is inspired by [qmd](https://github.com/tobi/qmd) but is not a fork
or a drop-in replacement: it's a from-scratch Python rewrite built around
a different set of tradeoffs — a shared Postgres/pgvector backend instead
of a per-project SQLite index, a pure HTTP LLM client instead of local
model loading, and a schema designed for eventual multi-user access
control instead of a single local user.

> [!NOTE]  
> Although inspired by `qmd`, this project has a completely different architecture based on a client-server model.
> The servers: `postgres` and `llm` can be on the local machine, containers, or remote. Setting them up is your task.
> If you want a product ready to use, the original `qmd` is your choice.

> [!CAUTION]  
> The project has been thoroughly [tested](devs/COVERAGE.md) and is fully functional; nonetheless, it must be considered in the alpha stage. 

## Features

- **Hybrid search** (`marq query`): automatic query expansion into
  lexical/semantic/hypothetical-document sub-queries, Reciprocal Rank
  Fusion across them, and LLM reranking of the fused candidates. Also
  accepts a hand-authored multi-line query document (`lex:`/`vec:`/
  `hyde:`/`intent:` lines) to skip automatic expansion entirely.
- **Full-text search** (`marq search`): Postgres `tsvector`/`ts_rank`,
  no LLM involved — fast, exact, supports quoted phrases and `-negation`.
- **Vector search** (`marq vsearch`): pgvector/HNSW similarity search,
  one physical table per embedding model, so switching models is
  additive (a new table), never destructive.
- **Retrieval**: `get` (by path or `#docid`, with line-range support)
  and `multi-get` (glob or comma-separated list), both line-numbered by
  default.
- **Collections**: any folder becomes a searchable, independently
  configurable collection (glob mask, optional pre-update command,
  default-query inclusion). Content is stored content-addressed by
  hash, shared across every collection, so re-indexing a mostly-unchanged
  folder (e.g. a second git worktree) is cheap.
- **Context**: attach human-written notes to a path or globally, surfaced
  alongside search results to disambiguate what a document actually is.
- **MCP server**: stdio or Streamable HTTP transport, four tools
  (`query`, `get`, `multi_get`, `status`) plus a `marq://collection/path`
  document resource, dynamic instructions built from live index state,
  and `/health` + `/query` (+ `/search` alias) REST endpoints alongside
  full MCP JSON-RPC on the HTTP transport.
- **Six-ish output formats**: `cli` (human-readable default), `json`,
  `csv`, `md`, `xml`, `files`, and `toon` (a compact LLM-oriented tabular
  encoding, see [toonformat.dev](https://toonformat.dev/)) — on every
  command that returns multiple results.
- **`marq bench`**: precision@k/recall/MRR/F1/latency across the bm25,
  vector, hybrid (no rerank), and full (reranked) backends against a
  fixture file.
- **`marq doctor`**: Postgres/pgvector/Alembic-migration/LLM-router
  health check.

## Requirements

- Python 3.13+
- A Postgres instance with the `vector` and `pg_trgm` extensions
  installable (superuser access needed once, for `CREATE EXTENSION`)
- An OpenAI-style HTTP endpoint for embeddings/chat/reranking (e.g. a
  local [llama.cpp server](https://github.com/ggml-org/llama.cpp) — see
  `llm-stack/README.md` for an optional local setup)

## Install

```bash
uv sync
```

This installs the `marq` console script into the project's virtualenv
(`uv run marq ...`, or activate the venv directly).

## Configuration

marq reads settings from environment variables (prefix `MARQ_`) or a
`.env` file in the current directory. Only `MARQ_POSTGRES_URL` is
required; see `.env.example` for a production-style template or
`.env.dev` for the local dev-container setup below.

| Variable                  | Default                            | Notes                                    |
|----------------------------|-------------------------------------|-------------------------------------------|
| `MARQ_POSTGRES_URL`       | *(required)*                       | `postgresql+psycopg://user:pass@host/db` |
| `MARQ_POSTGRES_SCHEMA`    | `qmd_py`                           | Postgres schema marq's tables live in    |
| `MARQ_LLM_BASE_URL`       | `http://localhost:8099`            | OpenAI-style embed/chat/rerank endpoint  |
| `MARQ_EMBED_MODEL`        | `bge-m3-q8_0`                      |                                            |
| `MARQ_GENERATE_MODEL`     | `qwen2.5-3b-instruct-q4_k_m`       | Used for query expansion                 |
| `MARQ_RERANK_MODEL`       | `qwen3-reranker-0.6b-q8_0`         |                                            |
| `MARQ_DEFAULT_USER_EMAIL` | `local@marq.local`                 | Single mocked user (see Architecture)    |
| `MARQ_LOG_LEVEL`          | `WARNING`                          | Silent on a healthy run; `-v`/`-vv` override |
| `MARQ_LOG_FILE`           | *(unset — stderr)*                 | Size-rotated file; never stdout          |

Running any command without `MARQ_POSTGRES_URL` set prints a short
"invalid or missing configuration" message naming the missing variable,
not a raw traceback.

## Quickstart

```bash
# 1. Local dev Postgres (pgvector/pgvector:pg16), or point MARQ_POSTGRES_URL
#    at your own instance instead.
cp .env.dev .env
podman-compose up -d          # or: docker compose up -d

# 2. Apply migrations.
uv run alembic upgrade head

# 3. Index a folder as a collection.
uv run marq collection add ~/notes --name notes --mask '**/*.md'

# 4. Generate vector embeddings for it (needed for vsearch/query, not search).
uv run marq embed

# 5. Search.
uv run marq search "connection pool timeout"
uv run marq query "how does the rate limiter handle bursts" --intent "web API design"
uv run marq get notes/rate-limiting.md
```

## Command reference

Read-only (no `MARQ_LLM_BASE_URL` needed except `query`/`vsearch`):

```
marq search <query>              Full-text keyword search (BM25, no LLM)
marq vsearch <query>             Vector similarity search (no reranking)
marq query <query>               Hybrid: expansion + RRF fusion + reranking (recommended)
marq deep-search <query>         Alias for `query`
marq get <file>[:from[:count]]   Get a document by path or #docid
marq multi-get <pattern>         Get multiple documents by glob or comma-separated list
marq ls [collection[/path]]      List collections, or files in a collection
marq status                      Show index status and collections
marq doctor                      Diagnose Postgres/pgvector/migrations/LLM-router health
marq bench <fixture.json>        Run search-quality benchmarks against a fixture file
```

Collection & context management (mutating):

```
marq collection add <path>                Add and index a collection
marq collection list                      List all collections with details
marq collection show <name>               Show collection details
marq collection remove <name>             Remove a collection by name
marq collection rename <old> <new>        Rename a collection
marq collection update-cmd <name> [cmd]   Set (or clear) the pre-update command
marq collection include <name>            Include a collection in default queries
marq collection exclude <name>            Exclude a collection from default queries
marq context add <path> <text>            Add context for a path ("/" for global)
marq context list                         List all contexts
marq context check                        Check for collections/paths missing context
marq context rm <path>                    Remove context for a path
marq update                               Re-index all collections
marq embed [-c <name>] [--model <slug>]   Generate vector embeddings for pending documents
marq cleanup                              Remove orphaned content/embeddings/documents
```

Agent integration:

```
marq mcp                          Start the MCP server (stdio transport)
marq mcp --http [--port N]        Start the MCP server (Streamable HTTP)
marq mcp --http --daemon          Start the HTTP server in the background
marq mcp stop                     Stop the background MCP daemon
marq skill show                   Print the bundled agent skill (bootstrap instructions)
marq skill install [--global]     Install the skill into .agents/skills/marq
marq skills list / get / path     Discover bundled skills generically
```

Every command supports `--help`. Search/retrieval commands that return
multiple results support `--format {cli,json,csv,md,xml,files,toon}`.

## MCP server

```bash
marq mcp                    # stdio, for editors/agents that spawn a subprocess
marq mcp --http --port 8181 # Streamable HTTP, for remote/persistent access
```

Exposes four tools (`query`, `get`, `multi_get`, `status`) and a
`marq://collection/path` document resource. The `query` tool accepts
either a plain-text `query` (auto-expanded) or explicit `searches`
(typed `lex`/`vec`/`hyde` sub-queries) — never both. Over HTTP, `/health`
and `/query` (+ `/search` alias) are also available as plain REST
endpoints alongside full MCP JSON-RPC at `/mcp`.

## Architecture

- **Storage**: Postgres + pgvector, one schema (`MARQ_POSTGRES_SCHEMA`,
  default `qmd_py`), managed by Alembic. Content is stored
  content-addressed by SHA-256 hash in one shared table, referenced by
  every collection's documents — indexing an unchanged file (a second
  worktree, a merged branch) costs nothing extra. Each embedding model
  gets its own physical table (`embeddings_<slug>`), so multiple models
  coexist and switching models never destroys data.
- **Search**: `tsvector`/`ts_rank` for BM25, pgvector HNSW for vector
  similarity, application-level Reciprocal Rank Fusion to combine
  multiple ranked lists, and an LLM reranking pass over the fused
  candidates.
- **LLM client**: a pure `httpx` HTTP client against an OpenAI-style
  endpoint — no local model loading, no GPU/device concept in the
  Python process itself.
- **ACL**: schema and call sites are structured for real multi-user
  access control (`Collection.owner_user_id`, a `CollectionGrant`
  table, a `can_access()` choke point called at every read/write path),
  but the actual check is mocked to always allow — the real use case
  today is a single local user. Swapping in a real check is additive:
  one function body, not a rearchitecture.
- **CLI**: `click`, one command (or command group) per file under
  `src/qmd_py/cli/commands/`.
- **MCP server**: the official `mcp` Python SDK's `MCPServer` (SDK 2.x).

## Development

```bash
uv sync --all-extras       # install dev + docs dependencies too
uv run pytest              # run the test suite (needs a reachable Postgres + LLM router)
uv run pytest -m "not integration"   # skip tests that need real Postgres/LLM
uv run ruff check --fix .
uv run mypy src alembic tests
uv run alembic check       # verify models match the latest migration
uv run zensical serve      # preview the docs site locally (docs/, zensical.toml)
```

Tests use a fresh, isolated Postgres schema per test run (see
`tests/conftest.py`) against the same instance `MARQ_POSTGRES_URL`
points at — `.env.dev` (copy to `.env`) starts a disposable
`pgvector/pgvector:pg16` container via `podman-compose up -d` (or
`docker compose up -d` — that service is runtime-agnostic) that matches
production's Postgres version exactly.

`tests/fixtures/sample-collection/` is a small, frozen, checked-in
multi-language fixture (`.md`/`.py`/`.js`/`.ts`) used by both the test
suite and `tests/fixtures/bench-sample-collection.json` (a benchmark
fixture) — no external data or SSH access needed to run realistic
multi-file tests.

CLI and MCP commands **are** end-to-end pytested. Both bind to a real,
process-global Postgres/LLM engine rather than the suite's per-test
isolated schema, so two conftest fixtures redirect that global at a
throwaway schema instead: `marq` (click's `CliRunner`) and `mcp_env` (a
real MCP `ClientSession` over the SDK's in-memory transport). See
`tests/test_cli.py` and `tests/test_mcp_server.py`.

Coverage is summarised in [`devs/COVERAGE.md`](devs/COVERAGE.md).
