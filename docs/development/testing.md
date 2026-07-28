# Setup & testing

```bash
uv sync --all-extras                 # install everything: dev, docs
uv run pytest                        # run the test suite
uv run pytest -m "not integration"   # skip tests needing real Postgres/LLM
uv run alembic upgrade head          # apply migrations
```

`uv sync` alone only installs marq's runtime dependencies — `pytest`/
`ruff`/`mypy` live under the `dev` optional-dependencies extra, and
`zensical`/`mkdocstrings`/`mkdocs-click` under `docs` (kept separate so
`uv sync --extra dev` for the fast day-to-day inner loop doesn't also
pull in the docs toolchain's Material-lineage dependency tree). Use
`--all-extras` unless you specifically want one or the other.

Local dev Postgres: `cp .env.dev .env && podman-compose up -d` starts a
disposable `pgvector/pgvector:pg16` container on `localhost:5433`,
version-matched to production. See [Configuration](../configuration.md).

## Testing conventions

**Fresh, isolated Postgres schema per test run.** Each test gets a
`qmd_test_<uuid>` schema (see `tests/conftest.py`), created against
whatever `MARQ_POSTGRES_URL` points at and dropped at teardown — not a
separate throwaway database, since that would need its own `vector`/
`pg_trgm` extension install.

**Tables are created from `SQLModel.metadata` directly, not via
Alembic**, in that isolated schema — these tests care about behavior,
not migration history.

**Integration tests** (`@pytest.mark.integration`) hit real Postgres and
the real LLM router. Run `pytest -m "not integration"` to skip them when
neither is reachable — most of the pure/logic-only modules (formatter,
scoring, vpath, skill discovery) have non-integration unit tests that
don't need either.

**Property-based tests live in `tests/test_properties.py`** (hypothesis),
and stay deliberately confined to it. They cover the pure, input-heavy
helpers — `extract_snippet`, `chunk_document`, `vpath`, and the
`build_ts_query` lex parser — where an invariant can actually be stated.
Every one of those was already at or near 100% line coverage; what a
property adds is *input* coverage, which is a different thing. The first
one written found a latent bound violation in `extract_snippet` that full
line coverage had never reached.

Two rules keep them cheap:

- **Derandomized.** `conftest.py` loads a `derandomize=True` profile, so a
  given commit always explores the same inputs. The rest of the suite is
  deterministic, and a property test that fails only on some runs would
  turn an unrelated change red for reasons its author can't reproduce.
  New counterexamples therefore appear when the code or the strategy
  changes — not spontaneously. `.hypothesis/` is build output and stays
  gitignored.
- **Sync, and off the async fixtures.** Hypothesis can't drive an async
  test (it raises `InvalidArgument`), and it health-checks function-scoped
  fixtures, which aren't reset between generated examples. The one
  property that needs Postgres — "whatever `build_ts_query` emits,
  `to_tsquery` accepts" — uses its own sync, autocommitting connection
  instead of the `session` fixture. Autocommit matters: a rejected
  tsquery aborts its transaction, so on a shared session every later
  example would fail with "current transaction is aborted" rather than
  its own verdict, and hypothesis would shrink toward the wrong input.

Do **not** reach for hypothesis against Postgres or the router generally.
That's where this project's real bugs have lived, and live verification —
not generated input — is what has actually found them.

**`tests/fixtures/sample-collection/`** is a small, frozen, checked-in
multi-language project (`.md`/`.py`/`.js`/`.ts`, the "Tasknote" example
used throughout these docs) — a manageable, always-available stand-in
for a real external collection, usable without SSH access or external
data. `tests/fixtures/bench-sample-collection.json` is a matching
benchmark fixture for `marq bench`.

**CLI and MCP commands are end-to-end pytested.** Two `conftest.py`
fixtures redirect `db/engine.py`'s process-global (`lru_cache`'d)
`get_engine()`/`get_session()` at a throwaway schema, which is what made
this possible: `marq` (sync, click's `CliRunner` — see
`tests/test_cli.py`, `test_cli_query.py`) and `mcp_env` (async, a real
MCP `ClientSession` over the SDK's in-memory transport — see
`tests/test_mcp_server.py`). Both patch `MARQ_POSTGRES_SCHEMA` and call
`reset_engine()`.

Two constraints come with that. CLI tests must be **sync**: every command
body ends in `cli/runtime.py`'s `asyncio.run()`, which refuses to start
inside a running loop. And `reset_engine()` has to run between
invocations in one process, since pooled connections don't survive their
event loop.

**Still verify live before calling a change done.** End-to-end tests
cover the command surface; they don't cover the real Postgres and router
your data actually lives on. Run `uv run marq <command>` against a
reachable instance — ideally both dev and production — cleaning up any
collection you added afterward.

This isn't a theoretical concern. Several real bugs in this project were
only caught by actually running the CLI/MCP server against real
infrastructure, never by a test:

- A Postgres `ts_rank` weight array out of the `[0, 1]` range it
  requires (a raw SQL error, only reachable by really running a search).
- A chunk sent to the LLM reranker that tokenized past the router's
  512-token batch-size cap — the chunker's char-based size estimate
  (~3 chars/token) undercounted dense code; only surfaced by actually
  reranking a real code file.
- An ACL-adjacent foreign-key cascade gap: reactivating a previously
  deactivated document, and cleaning up orphaned content referenced only
  by an *inactive* document, both crashed with a raw `IntegrityError`/
  `ForeignKeyViolation` the first time they were exercised through a
  real `mv`/`update`/`mv`/`update` CLI sequence — not caught by any
  fixture-seeded unit test, because the fixtures never happened to
  reproduce that exact sequence.

None of these were exotic edge cases — they were the *normal* path,
just not one any test had walked yet. Live verification against real
infrastructure isn't a formality in this project; it's the step that's
actually caught every one of these bugs.
