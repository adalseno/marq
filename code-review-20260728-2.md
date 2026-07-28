# Code review — marq (qmd-py), fourth pass

*Senior-developer review, 2026-07-28, following up on
`code-review-20260728.md` after all nine of its actionable findings were
implemented. Reviewed at commit `2c53633`. Scope: fresh-eyes pass over
the whole codebase with emphasis on the newly landed changes (logging,
skip paths, the daemon rework) and their interactions — do the fixes
introduce new bugs or uncovered edge cases — plus the requested final
verdict: is the project ready for an alpha deployment?*

*Resolved same day in two commits: findings 1–3 in `b2e54a1`, then the
three minor polish items in `1c43b5c`, each with a regression test.
Nothing from this pass is left open.*

## Outcome

Three findings and three polish items, all closed:

| # | Finding | Resolution |
|---|---------|------------|
| 1 | Oversize cap counts characters, tsvector cap is bytes; `collection add` still strands a committed empty collection on a reindex crash | **Fixed** `b2e54a1` — `MAX_INDEXABLE_CHARS` became `MAX_INDEXABLE_BYTES`, measured against the UTF-8 encoding; `_collection_add_impl` now catches `OSError`/`SQLAlchemyError` around the initial reindex, removes the just-created collection, and exits 1 with "Collection '…' was not created" |
| 2 | Empty-file skip warns on healthy input | **Fixed** `b2e54a1` — demoted to DEBUG; docstring realigned with the logging that actually exists |
| 3 | `pytest>=8.0` floor predates non-propagating-logger capture | **Fixed** `b2e54a1` — floor bumped to `pytest>=8.4`, with the reason recorded next to the constraint |
| P1 | A failed rerank still logs `reranked=yes` | **Fixed** `1c43b5c` — the except branch sets `timing["reranked"] = "failed"` |
| P2 | REST `/query` accepts a negative `limit`, dropping results off the end | **Fixed** `1c43b5c` — clamped with `max(0, limit)` in `_run_query_search`, which covers the `query` tool as well as the REST route |
| P3 | `mcp-stdio.log` truncation wipes a crashed daemon's traceback | **Fixed** `1c43b5c` — the previous file is rotated to `mcp-stdio.log.1` before truncation |

## Findings in detail

1. **The oversize-skip fix from the third review had a live-confirmed
   gap: the cap counted *characters*, but Postgres's tsvector limit is
   *bytes*.** Confirmed with a read-only probe against the disposable
   Postgres: 868,889 characters of distinct Cyrillic tokens — well under
   the 1,000,000-character cap — is 1,268,889 bytes of UTF-8 and raises
   `ProgramLimitExceeded: string is too long for tsvector (1579800
   bytes, max 1048575 bytes)`. Any multibyte-heavy, distinct-token file
   (Cyrillic, Greek, CJK, emoji) could slip through the cap and still
   abort the reindex.

   The blast radius split exactly along the line the third review drew:
   `update` survived it (the per-collection `SQLAlchemyError` catch
   contained it), but **`collection add` commits the collection row
   before reindexing**, so the crash still left a committed, empty
   collection behind — and the retry was told "already exists. Use a
   different name", which is exactly wrong.

   Fixed on both ends: the cap is now `MAX_INDEXABLE_BYTES`, compared
   against `len(content.encode("utf-8"))` (the byte length is what
   Postgres enforces; encoding cost is noise next to the SHA-256 hashing
   that already touches every byte), and `_collection_add_impl` wraps
   the initial reindex — on `OSError`/`SQLAlchemyError` it rolls back,
   removes the collection it just created, and exits 1, so a retry with
   the same name actually retries. Regression tests:
   `test_reindex_oversize_cap_measures_bytes_not_characters` (a
   sub-cap-chars / over-cap-bytes Cyrillic file against real Postgres)
   and `test_collection_add_undoes_the_collection_when_indexing_fails`
   (a failing reindex, then a successful retry under the same name).

2. **The empty-file skip logged WARNING on healthy input, breaking the
   log's own "silent on a healthy run" contract.** Empty `__init__.py`
   files are completely normal in a code collection, and placeholder
   notes are normal in a markdown vault — yet every `marq update` run
   re-warned about each one, forever, into the rotating daemon log under
   cron. Notably, the third review spec'd WARNING for
   unreadable/non-UTF-8/oversize only; empty files were swept in during
   implementation. Demoted to DEBUG (an empty file is a state, not a
   degrade), with the reasoning recorded at the call site. The
   `reindex_collection` docstring — which still claimed these files were
   "skipped silently, not reported" — now describes the per-case levels.
   The per-skip logging test now asserts levels, not just presence: the
   unreadable and oversized skips must WARN, the empty skip must not.

3. **The `pytest>=8.0` floor was too low for the new logging tests.**
   `create_mcp_server()` sets `qmd_py.propagate = False` process-wide
   (the FastMCP `basicConfig` double-logging fix), and `caplog` cannot
   capture through a non-propagating logger — verified empirically:
   flipping `propagate` mid-test makes capture fail. It works in this
   suite only because pytest ≥8.4 attaches its capture handler directly
   onto loggers that are already non-propagating at test setup
   (confirmed against the installed 9.1.1's `catching_logs`). On pytest
   8.0–8.3 the `caplog` assertions in `test_hybrid.py` and
   `test_indexing.py` would pass only by the alphabetical accident of
   running before any test that constructs the MCP server. One-line fix:
   `pytest>=8.4`, with a comment explaining why the floor is where it
   is.

## Minor polish (filed as optional, then done anyway)

All three were one-liners with a cheap test each, so they were taken
rather than deferred — together in `1c43b5c`:

- **P1.** `hybrid.py` set `timing["reranked"] = "yes"` *before* the
  rerank attempt, so a failed rerank's INFO line still said
  `reranked=yes` — contradicting the WARNING logged immediately above
  it. The except branch now sets `"failed"`. The existing rerank
  fallback test was widened from WARNING to INFO capture and asserts the
  `query:` timing line reads `reranked=failed`.
- **P2.** REST `/query` accepted a negative `limit`: `_is_int(-5)`
  passes the boolean fence, and `results[:-5]` then silently drops
  results off the *end* — the caller asked for fewer and got "all but
  the last five". The clamp went into `_run_query_search` rather than
  the route, because the `query` MCP tool's `limit` field carries no
  `ge` constraint either and had the identical hole; one `max(0, limit)`
  closes both. `test_rest_query_clamps_a_negative_limit` drives the real
  ASGI app with `limit: -1` against two matching documents and expects
  zero results, not one.
- **P3.** `mcp-stdio.log` was truncated at each daemon *start*, wiping a
  crashed daemon's traceback at the exact moment the user retried the
  start to read it. The file is now rotated to `mcp-stdio.log.1` first —
  the same one-backup shape as the rotating handler on the daemon's own
  log, so the evidence survives while growth stays bounded at two files.
  Truncate-on-*successful*-start was the other option in the note, but
  "successful" is not knowable at `Popen` time; the rotation is
  deterministic instead. `test_start_daemon_keeps_the_previous_stdio_log_as_a_backup`
  covers it behind a stubbed `Popen`, which also lifted
  `cli/commands/mcp.py` from 36% to 50%.

## What held up under scrutiny

Chased and cleared, for the record: the `/query`/`/search` route
registration around the new `_with_request_id` decorator (registered
explicitly, correct); the shared `LlmClient` lifecycle (`aclose` via the
FastMCP lifespan hook, concurrency-safe); the request-id filter living
on the handler (correct — propagation semantics verified); the daemon's
`model_fields_set` level defaulting and env plumbing; the
segment-boundary `endswith` fix (the full-path form still matches thanks
to the `//` in `marq://`); `setup_logging` idempotence vs. the `-v`
force path; the `_is_int` boolean fence; and the state-dir `mkdir`. The
TOCTOU fix, the update-command timeout, and the public-bind warning all
carry proper regression tests. The watchlist items from the second and
third reviews stay where those reviews filed them: scale-dependent debt
that has not come due.

## Deployment verdict (alpha)

**Ready.** 424 tests at 93% coverage with genuine end-to-end CLI and MCP
coverage, clean strict typing and linting, every silent degrade path now
leaves log evidence, and the daemon has rotation and correlation ids.
Four review cycles have converged: this pass found no new bug class —
only a residual edge inside a previous fix and a log-level calibration,
both now closed. The known constraints — mocked ACL, unauthenticated
HTTP transport — are documented, warned about at startup, and
appropriate for the actual deployment shape (single local user, loopback
bind). The watchlist is scale-dependent debt, correctly deferred.

## Verification

- Pre-change baseline at `2c53633`: full suite **420 passed** (including
  the 145 integration tests against the local disposable Postgres and
  the real router); `ruff check`, `mypy src alembic tests`, and
  `zensical build --strict` all clean.
- Finding 1 was **reproduced live** as a read-only SQL probe against the
  disposable Postgres (`localhost:5433`): a pure
  `SELECT to_tsvector(...)` on 868,889 chars / 1,268,889 UTF-8 bytes of
  distinct Cyrillic tokens → `string is too long for tsvector`. No marq
  tables touched, nothing to clean up.
- Finding 2 was demonstrated with a scratch test simulating a healthy
  Python collection (an empty `__init__.py` warning on every run);
  finding 3 with a scratch test proving `caplog` misses records through
  a mid-test `propagate = False`, plus inspection of the installed
  pytest's `catching_logs`.
- Post-fix (`b2e54a1`): `tests/test_indexing.py` plus the two
  `collection add` CLI tests — **13 passed** against real Postgres;
  `ruff` and `mypy` clean; full suite re-run **422 passed** (including
  integration) with the fixes in place.
- After the three polish items (`1c43b5c`): full suite **424 passed** at **93%**
  coverage (`uv run pytest --cov=qmd_py`, integration included),
  `ruff check` and `mypy src alembic tests` clean, `zensical build
  --strict` reporting no issues. `devs/COVERAGE.md` refreshed to match.
  No mutating command was run against real data and no collection was
  created outside the tests' throwaway schema, so there was nothing to
  clean up.
