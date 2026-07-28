# Configuration

marq reads settings from environment variables (prefix `MARQ_`) or a
`.env` file in the current directory (via `pydantic-settings`). Only
`MARQ_POSTGRES_URL` is required — everything else has a default.

Running any command without `MARQ_POSTGRES_URL` set prints a short,
actionable error instead of a raw traceback:

```console
$ marq doctor
marq: invalid or missing configuration
  MARQ_POSTGRES_URL: Field required

Set these as environment variables or in a .env file in the current directory.
```

This comes from `cli/main.py`'s `main()`, which wraps the whole CLI in
a `try/except` around pydantic's `ValidationError` — `get_settings()` is
called lazily from deep inside individual commands' (often async)
bodies, not once up front, so this outermost wrapper is the one place
that reliably catches it regardless of which command triggered it.

## Which file to start from

- **`.env.dev`** (copy to `.env`) — the local dev-container setup: a
  disposable `pgvector/pgvector:pg16` Postgres on `localhost:5433` with
  throwaway credentials (safe to commit, not secrets), version-matched
  to production. Use this for day-to-day development and running the
  test suite.
- **`.env.example`** — a template for pointing marq at your own
  Postgres/LLM instance directly (production, or a real-server
  live-parity check). Copy it, fill in your own host/credentials.

## Variables

### `MARQ_POSTGRES_URL`

*(required, no default)* — `postgresql+psycopg://user:pass@host/db`.

The only setting with no default, since there's no sensible one: marq
has no bundled database, and every other setting can be inferred or
defaulted around a working Postgres connection.

### `MARQ_POSTGRES_SCHEMA`

Default: `qmd_py`.

The Postgres schema marq's tables live in, inside whatever database
`MARQ_POSTGRES_URL` points at — not a separate database. This lets marq
coexist in the same Postgres instance as unrelated services (including,
historically, the original TS `qmd` reference implementation, which
used `public`) without their tables colliding. The default value still
reads `qmd_py` rather than `marq` — that's the Python package name
(`qmd_py`), which stayed as-is during the CLI/MCP rebrand since it's an
internal storage detail, not something you normally need to change. See
[Architecture › Storage](architecture.md#storage) for why a single
schema (not schema-per-collection or similar) was the right call.

### `MARQ_LLM_BASE_URL` {: #marq_llm_base_url }

Default: `http://localhost:8099` — a placeholder, not a router that
exists. Set this to wherever yours actually runs.

An OpenAI-style HTTP endpoint serving embeddings (`/v1/embeddings`),
chat completions (`/v1/chat/completions`, used for query expansion),
and reranking (`/rerank`) — e.g. a
[llama.cpp server](https://github.com/ggml-org/llama.cpp). marq's LLM
client (`llm/client.py`) is a pure `httpx` HTTP client with no local
model loading concept at all, so this is the only thing that needs to
be reachable for `vsearch`/`query`/`embed`/`doctor`'s router check to
work. See `llm-stack/README.md` for running one locally instead of
depending on a shared instance.

### `MARQ_EMBED_MODEL`

Default: `bge-m3-q8_0`.

The embedding model slug used for `marq embed` and vector/hybrid search.
Each embedding model gets its own physical Postgres table
(`embeddings_<slug>`), so changing this doesn't destroy any previously
generated embeddings under a different model — see
[Architecture › Storage](architecture.md#storage).

### `MARQ_GENERATE_MODEL`

Default: `qwen2.5-3b-instruct-q4_k_m`.

The chat-completion model used for `marq query`'s automatic query
expansion (turning your text into typed `lex`/`vec`/`hyde` sub-queries).
Not used at all if you write the structured `lex:`/`vec:`/`hyde:` query
document yourself — see
[Search & query](commands/search-and-query.md).

### `MARQ_RERANK_MODEL`

Default: `qwen3-reranker-0.6b-q8_0`.

The reranking model `marq query` uses to score fused candidates against
your query, unless you pass `--no-rerank`.

### `MARQ_DEFAULT_USER_EMAIL`

Default: `local@marq.local`.

Identifies the single mocked local user marq creates on first use — see
[Architecture › ACL](architecture.md#acl) for why this exists as a real
row (not just an abstract concept) even though there's only ever one of
it today.

### `MARQ_LOG_LEVEL`

Default: `WARNING`.

One of `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`. An unrecognised
value falls back to `WARNING` rather than failing the command.

The default is **silent on a healthy run** — that's what makes the log
worth reading: a non-empty log means something actually degraded (query
expansion or reranking fell back, a file was skipped at index time, an
update command failed, a request was rejected).

`INFO` adds one line per unit of work with timings and counts — a query's
sub-queries/candidates/results, a reindex's per-bucket counts, an embed
run's documents and chunks, one line per REST request. It is the
sensible level for a long-running server, and `marq mcp --http --daemon`
selects it automatically unless you set this variable yourself.

`DEBUG` adds the full mechanism, **including content**: query texts,
expansion variants, RRF weight tables. It also un-silences `httpx`,
`sqlalchemy.engine` and the MCP SDK's own loggers, which are pinned at
`WARNING` at every other level.

For one-off debugging, the CLI's `-v` (INFO) and `-vv` (DEBUG) flags do
the same thing without touching the environment:

```sh
marq -v query "how does auth work"
```

### `MARQ_LOG_FILE`

Default: unset (log to stderr).

Path to a size-rotated log file (5 MB, 3 backups). Logging **never** goes
to stdout: the CLI's stdout carries parseable output (`--format
json/csv`) and the MCP stdio transport *is* JSON-RPC over stdout, so a
stray log line would corrupt either one.

`marq mcp --http --daemon` defaults this to `~/.cache/marq/marq.log`,
since the caller's stderr is gone the moment the command returns. Any
output the daemon writes outside the logging system (a startup traceback,
a library printing directly to stderr) lands in
`~/.cache/marq/mcp-stdio.log`. That file is rotated at each start — the
previous run moves to `mcp-stdio.log.1` — so a crashed daemon's traceback
survives the restart you do to investigate it, while growth stays bounded
at two files.

!!! note "What gets logged, and what never does"

    At `WARNING` and `INFO`, lines carry *shapes* — counts, lengths,
    paths, durations, exception types — and never *content*: no query
    text, no document bodies, no snippets. The index holds whatever you
    pointed it at, and a log file outlives the collection
    (`marq collection remove` deletes documents, not log lines). Content
    appears only at `DEBUG`, which is opt-in.
