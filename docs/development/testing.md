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

**`tests/fixtures/sample-collection/`** is a small, frozen, checked-in
multi-language project (`.md`/`.py`/`.js`/`.ts`, the "Tasknote" example
used throughout these docs) — a manageable, always-available stand-in
for a real external collection, usable without SSH access or external
data. `tests/fixtures/bench-sample-collection.json` is a matching
benchmark fixture for `marq bench`.

**CLI and MCP commands are deliberately *not* end-to-end pytested.**
They call `db/engine.py`'s real, process-global (`lru_cache`'d)
`get_engine()`/`get_session()`, bound to whatever `MARQ_POSTGRES_URL`
actually resolves to at runtime — not the test suite's per-test
isolated schema. Retrofitting that (redirecting the real engine to an
isolated schema via monkeypatching, working around the cross-event-loop
connection-pool hazards that come with it) was considered and rejected:
it would fight this project's existing design for uncertain benefit.
**Verify these live instead**: run `uv run marq <command>` against a
real reachable Postgres + LLM router — ideally both a dev instance and
production before calling a change done, cleaning up any collection you
added afterward.

This isn't a theoretical concern. Several real bugs in this project were
only caught by actually running the CLI/MCP server against real
infrastructure, never by a unit test:

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
