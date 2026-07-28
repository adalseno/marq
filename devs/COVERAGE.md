# Test coverage

Snapshot of the test suite's coverage of `src/qmd_py`, committed so the
project's real state is visible without cloning and running anything.

**441 tests, 93% line coverage** — of those, **292 need no infrastructure
at all** and run in about thirty seconds (`uv run pytest -m "not integration"`).
The remaining 149 are integration tests hitting a real Postgres and a real
LLM router; see [CLAUDE.md](../CLAUDE.md) for the testing conventions.

Seventeen of those are property-based (`tests/test_properties.py`,
hypothesis). They barely move the line-coverage numbers below — every
function they cover was already at or near 100% — because what they add is
*input* coverage, not reachability. They are also the bulk of that thirty
seconds: about half, against ~17s for everything else. `max_examples` is
capped at 100 to keep that ratio defensible.

> [!NOTE]
> A hand-refreshed snapshot, so it can lag the code. Regenerate the table
> with:
>
> ```sh
> uv run pytest --cov=qmd_py --cov-report=term-missing
> uv run coverage report --format=markdown
> ```
>
> For a browsable, line-by-line view locally, add `--cov-report=html` and
> open `htmlcov/index.html`. That report is a build artifact and stays
> gitignored — publishing it is a job for CI once the project moves to
> GitHub Actions/Pages.

## Where the gaps are

The uncovered code is concentrated and deliberate rather than accidental:

- **`cli/commands/mcp.py` (50%)** — daemon start/stop. `_start_daemon`'s
  bookkeeping (stdio-log rotation, pid file) is now unit-tested behind a
  stubbed `Popen`; what's left needs really spawning and killing
  background processes (done by hand: see the third review's resolution
  record).
- **`cli/commands/bench.py` (47%)** — the thin click wrapper around the
  benchmark runner; `bench.py` itself is at 93%.
- **`mcp/server.py` (91%)** — the remainder is the REST routes, which only
  get registered under `marq mcp --http`. Their malformed-input handling
  is now covered (a `/query` 400 regression test driving the real ASGI
  app); what's left is the success paths, which need a live LLM router.
- **`cli/commands/doctor.py` (79%)** — the per-check failure branches,
  which need a deliberately broken Postgres or router to reach.

Everything else is at or above 84%, and the search/storage core
(`store/`, `search/`, `llm/`, `log.py`, `cli/formatter.py`,
`cli/snippet.py`, `config.py`, `db/`) is at 93–100%.

## Per-module

| Name                                     |    Stmts |     Miss |   Cover |
|----------------------------------------- | -------: | -------: | ------: |
| src/qmd\_py/\_\_init\_\_.py              |        0 |        0 |    100% |
| src/qmd\_py/auth.py                      |       23 |        0 |    100% |
| src/qmd\_py/bench.py                     |      219 |       15 |     93% |
| src/qmd\_py/cli/\_\_init\_\_.py          |        0 |        0 |    100% |
| src/qmd\_py/cli/commands/\_\_init\_\_.py |        0 |        0 |    100% |
| src/qmd\_py/cli/commands/bench.py        |       30 |       16 |     47% |
| src/qmd\_py/cli/commands/doctor.py       |      108 |       23 |     79% |
| src/qmd\_py/cli/commands/mcp.py          |       92 |       46 |     50% |
| src/qmd\_py/cli/commands/query.py        |      104 |       10 |     90% |
| src/qmd\_py/cli/commands/read.py         |      218 |       20 |     91% |
| src/qmd\_py/cli/commands/skill.py        |      104 |        8 |     92% |
| src/qmd\_py/cli/commands/write.py        |      321 |       52 |     84% |
| src/qmd\_py/cli/formatter.py             |      243 |        0 |    100% |
| src/qmd\_py/cli/main.py                  |       53 |        3 |     94% |
| src/qmd\_py/cli/runtime.py               |       11 |        0 |    100% |
| src/qmd\_py/cli/snippet.py               |       68 |        0 |    100% |
| src/qmd\_py/config.py                    |       28 |        0 |    100% |
| src/qmd\_py/db/\_\_init\_\_.py           |        0 |        0 |    100% |
| src/qmd\_py/db/engine.py                 |       18 |        0 |    100% |
| src/qmd\_py/db/models.py                 |       66 |        0 |    100% |
| src/qmd\_py/db/result.py                 |        6 |        0 |    100% |
| src/qmd\_py/llm/\_\_init\_\_.py          |        0 |        0 |    100% |
| src/qmd\_py/llm/client.py                |       55 |        0 |    100% |
| src/qmd\_py/log.py                       |       57 |        0 |    100% |
| src/qmd\_py/mcp/\_\_init\_\_.py          |        0 |        0 |    100% |
| src/qmd\_py/mcp/server.py                |      300 |       28 |     91% |
| src/qmd\_py/search/\_\_init\_\_.py       |        0 |        0 |    100% |
| src/qmd\_py/search/\_acl.py              |       16 |        0 |    100% |
| src/qmd\_py/search/fts.py                |      167 |        2 |     99% |
| src/qmd\_py/search/hybrid.py             |      269 |        2 |     99% |
| src/qmd\_py/search/vector.py             |      147 |       11 |     93% |
| src/qmd\_py/skills/\_\_init\_\_.py       |       67 |        6 |     91% |
| src/qmd\_py/store/\_\_init\_\_.py        |        8 |        0 |    100% |
| src/qmd\_py/store/\_common.py            |       50 |        0 |    100% |
| src/qmd\_py/store/cleanup.py             |       22 |        1 |     95% |
| src/qmd\_py/store/collection.py          |       64 |        0 |    100% |
| src/qmd\_py/store/context.py             |       63 |        1 |     98% |
| src/qmd\_py/store/documents.py           |       36 |        0 |    100% |
| src/qmd\_py/store/indexing.py            |      116 |        3 |     97% |
| src/qmd\_py/store/retrieval.py           |      186 |        2 |     99% |
| src/qmd\_py/vpath.py                     |       12 |        0 |    100% |
| **TOTAL**                                | **3347** |  **249** | **93%** |
