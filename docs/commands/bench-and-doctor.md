# Bench & doctor

## `marq bench` — search-quality benchmarks

Runs a set of test queries through four backends and reports
precision@k, recall (overall and at 1/3/5), MRR, F1, and latency for
each:

| Backend | What it runs |
|---|---|
| `bm25` | `search` only |
| `vector` | `vsearch` only |
| `hybrid` | Full `query` pipeline, RRF-fused, **no** reranking |
| `full` | Full `query` pipeline, RRF-fused, **with** reranking |

```console
$ uv run marq bench tests/fixtures/bench-sample-collection.json -c tasknote
...
Summary:
----------------------------------------------------------------------
  bm25     P@k= 0.500 R@1= 0.333 R@3= 0.500 R@5= 0.667 MRR= 0.458 F1= 0.500 Avg=19ms
  hybrid   P@k= 0.917 R@1= 0.500 R@3= 0.917 R@5= 1.000 MRR= 0.750 F1= 0.944 Avg=2757ms
  full     P@k= 1.000 R@1= 0.667 R@3= 1.000 R@5= 1.000 MRR= 0.833 F1= 1.000 Avg=3706ms
  vector   P@k= 0.917 R@1= 0.667 R@3= 0.917 R@5= 1.000 MRR= 0.833 F1= 0.944 Avg=98ms
```

(`tests/fixtures/bench-sample-collection.json` is checked into the repo
and pairs with the `tests/fixtures/sample-collection/` fixture used
throughout these docs — index it as `tasknote` per the
[Quickstart](../quickstart.md) and this command runs as-is.)

`bm25` is fastest but weakest on anything conceptual; `full` (the
default `query` pipeline) is slowest but strongest — exactly the
tradeoff [Search & query](search-and-query.md) describes, quantified.

### Fixture format

A fixture is a JSON file with a top-level `description`, `version`, an
optional default `collection`, and a `queries` array
(`BenchmarkQuery` in `src/qmd_py/bench.py`):

```json
{
  "description": "...",
  "version": 1,
  "collection": "sample",
  "queries": [
    {
      "id": "exact-http-api",
      "query": "HTTP API",
      "type": "exact",
      "description": "Direct keyword match - 'HTTP API' appears verbatim in api.js",
      "expected_files": ["src/api.js"],
      "expected_in_top_k": 1
    }
  ]
}
```

- `expected_files` — paths (relative to the collection) that should
  appear somewhere in the results.
- `expected_in_top_k` — how many of `expected_files` should appear in
  the *first* `k` results specifically (used for precision@k; recall is
  computed against the whole returned result set, not just the top-k).
- `type` is free-form and only used for your own grouping/labeling — not
  interpreted by the scorer.

`-c/--collection` on the command line overrides the fixture's own
`collection` field. `--format json` gives the full per-query breakdown
(`marq bench <fixture> --format json`) instead of the summary table.

## `marq doctor` — health check

```console
$ uv run marq doctor
marq doctor

Postgres schema: qmd_py
LLM router: http://ubuserver.internal:8099

✓ Postgres connectivity: PostgreSQL 16.14 (Debian 16.14-1.pgdg12+1) on x86_64-pc-linux-gnu, ...
✓ pgvector extension: 0.8.5
✓ Migrations: up to date (2c4074f0444c)
✓ Collections: 1 configured
✓ Vector index: bge-m3-q8_0 up to date
✓ LLM router: reachable, 8 model(s) available
✓   embed model: bge-m3-q8_0
✓   generate model: qwen2.5-3b-instruct-q4_k_m
✓   rerank model: qwen3-reranker-0.6b-q8_0

Effective configuration:
  MARQ_POSTGRES_URL       postgresql+psycopg://qmd:***@localhost:5433/qmd
  MARQ_POSTGRES_SCHEMA    qmd_py
  MARQ_LLM_BASE_URL       http://ubuserver.internal:8099
  MARQ_EMBED_MODEL        bge-m3-q8_0
  MARQ_GENERATE_MODEL     qwen2.5-3b-instruct-q4_k_m
  MARQ_RERANK_MODEL       qwen3-reranker-0.6b-q8_0
  MARQ_DEFAULT_USER_EMAIL local@marq.local
```

Each check, and what it means when it fails:

- **Postgres connectivity** — can't connect at all; check
  `MARQ_POSTGRES_URL`.
- **pgvector extension** — the `vector` extension isn't installed in the
  database; needs `CREATE EXTENSION vector` as a superuser, once.
- **Migrations** — compares the schema's current Alembic revision
  against the latest one in `alembic/versions/`; a mismatch means
  `alembic upgrade head` hasn't been run. Only works from a source
  checkout (needs `alembic.ini`) — reports a soft warning, not a
  failure, when run from an installed package elsewhere.
- **Collections** — zero configured isn't a failure, just a nudge to run
  `collection add`.
- **Vector index** — reports how many active documents still need
  embedding for the configured `MARQ_EMBED_MODEL`.
- **LLM router** — unreachable, or reachable but missing one of the
  three configured models (embed/generate/rerank) by exact id.

The `Effective configuration` block always redacts the password portion
of `MARQ_POSTGRES_URL` (`user:***@host`) before printing it — safe to
paste into a bug report or a chat.
