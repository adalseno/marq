# Code review — marq (qmd-py)

*Senior-developer review, 2026-07-27. Scope: test coverage, edge cases,
architecture soundness, and coupling. Reviewed at commit `be89ddf`.*

*Resolved 2026-07-27 across twelve commits, `83ff6c2`…`8c56980`. Every
finding and every priority item is closed; this file is kept as the record
of what was found, what was done about it, and what was deliberately left
alone.*

## Outcome

|                          | Before  | After   |
|--------------------------|---------|---------|
| Line coverage            | 70%     | **90%** |
| Tests passing            | 181     | **377** |
| Runnable without infra   | 119     | **250** (~1s) |
| Largest module           | `store.py`, 1076 lines | `store/` package, 8 modules |
| `hybrid_query` signature | 14 params | 6 |

Three bugs were found *by* the new tests, after the review — they are
listed under "Found during the work" below.

## TL;DR (original)

The architecture is solid — genuinely well above "first draft" quality. The
service-facade design, the ACL choke-point structure, and the
content-addressed storage are all sound, and the code comments documenting
live-caught failure modes are exemplary. Real coverage is **70% with the full
suite (181 tests passing, integration included)**, not the 49% the unit-only
run suggests. The coverage gap is concentrated in exactly one place — the
CLI/MCP command layer — and it exists because of the one real coupling
problem in the codebase: the process-global `lru_cache`'d engine/settings
singletons. Fix that one thing and both the coverage ceiling and the coupling
concern largely disappear. The review also found one small real bug and some
dead code.

That diagnosis held up. The engine singleton was indeed the whole blocker:
one fixture pair unlocked ~700 statements of CLI/MCP coverage, and the
CLI/MCP layer went from 28–42% to 77–92%.

## Concrete findings (bugs / dead code)

1. ✅ **`_expand_braces` only expanded the first brace group** — `83ff6c2`.
   `_expand_braces('{src,docs}/**/*.{md,py}')` returned
   `['src/**/*.{md,py}', 'docs/**/*.{md,py}']`; the second `{...}` survived
   as a literal, so glob matched nothing and the collection **silently
   indexed zero files**. Single-group patterns (the documented
   `**/*.{py,md,json}` case) worked, which is why nothing caught it.
   Fixed by recursing until no group remains, with dedup (nested groups can
   produce the same expansion twice) and a 1000-expansion cap (groups
   multiply; each expansion is its own filesystem walk). Nine tests,
   including one on `_discover_files` against a real directory tree — the
   layer `reindex_collection` actually calls.

2. ✅ **`validate_lex_query` / `validate_semantic_query` were dead code** —
   `734d406`. Wired rather than deleted: the messages are actionable and
   were clearly written on purpose. A new `validate_typed_queries()` helper
   is applied at the three places a caller spells sub-queries out — the CLI's
   multi-line query document, the MCP `query` tool's `searches`, and the
   REST `/query` alias. CLI exits 1, MCP returns `isError`, REST returns 400.
   **Deliberately not** applied to `expand_query()`'s LLM-generated variants
   or to plain single-line queries: a stray `-term` there is the model's
   doing, not a user mistake worth failing a search over. Lex negation stays
   legal; all three exemptions are pinned by regression tests.

3. ✅ **Docid prefix matching was nondeterministic on collision** —
   `1ffc0d5`. `_active_document_refs` had no `ORDER BY`, so which of two
   documents sharing a 6-char docid won came down to Postgres's row order.
   Now ordered by `(collection name, path)`, which also makes the suffix
   match and glob output stable. Took the "order deterministically" option
   rather than "detect ambiguity and error".

4. ✅ **The rerank pass had no graceful-degrade path** — `1ffc0d5`,
   completed in `7711455`. The first pass only hardened
   `LlmClient.rerank` against out-of-range indices from the router (a large
   index raised `IndexError`; a negative one silently scored the wrong
   document from the end) and established that the `zip(..., strict=True)`
   is safe, since `rerank` always returns one score per document.
   That missed the actual point, caught on re-reading this file: the
   `/rerank` and `/tokenize` calls were unguarded, so a 500 or timeout
   **failed the entire query** even though retrieval and RRF fusion had
   already produced a good candidate list — the user lost every result to a
   failure in the optional refinement step. The `skip_rerank` block is now
   `_rrf_only_results()` and serves both callers. Verified live by pointing
   `MARQ_RERANK_MODEL` at a nonexistent model.

5. ✅ **Minor** — `1ffc0d5`. `hash_content` is sync now (pure CPU; it was
   `async` only because the TS original returns a Promise). `multi_get`'s
   comma-vs-glob heuristic was **documented rather than changed**: the two
   forms don't combine, so `"a.md,b*.md"` is one glob with a literal comma
   and matches nothing. That is the TS reference's behavior, kept for
   parity and now pinned by a test so it can't drift silently.

## Test coverage

✅ **The structural fix** — `2416348`. Two conftest fixtures redirect
`db/engine.py`'s process-global, `lru_cache`'d `get_engine()` at a throwaway
schema: `marq` (sync, click `CliRunner`) and `mcp_env` (async, driving a real
MCP `ClientSession` over the SDK's in-memory transport, so
`initialize`/`tools/call`/`resources/read` go through the actual JSON-RPC
handlers). Both patch `MARQ_POSTGRES_SCHEMA` and clear the caches through a
new `reset_engine()` helper.

Two constraints, both documented in code because neither is guessable:
CLI tests **must be sync** (every command body ends in `asyncio.run()`,
which refuses to start inside a running loop), and `reset_engine()` must run
**before each CLI invocation** — `asyncio.run()` closes its loop on exit and
pooled connections don't survive it, so a second command in one test would
otherwise get connections bound to a dead loop.

| Module                   | Before | After |
|--------------------------|--------|-------|
| `cli/commands/write.py`  | 28%    | 77%   |
| `cli/commands/read.py`   | 33%    | 90%   |
| `cli/commands/query.py`  | 34%    | 90%   |
| `cli/commands/skill.py`  | 30%    | 92%   |
| `cli/commands/doctor.py` | 38%    | 79%   |
| `mcp/server.py`          | 42%    | 79%   |
| `llm/client.py`          | 87%    | 100%  |
| `cli/snippet.py`         | 90%    | 100%  |

✅ **`LlmClient` offline tests** — `6936a4a`. Thirty tests via
`httpx.MockTransport`: the index-keyed reordering contracts in `embed` and
`rerank` (the router answers by index, not input order — getting this wrong
gives every chunk the wrong embedding), the pre-filled zero scores, and
behavior on 500 / read timeout / connect error / unparseable body. Those
exception types are a real contract: `expand_query` catches exactly
`(httpx.HTTPError, KeyError, ValueError, TypeError)`, so they are now pinned
on both sides. `LlmClient` gained an optional `transport` parameter, httpx's
standard injection seam, so tests never reach into a private attribute.

✅ **Edge cases** — `3985a35`, plus `147b4e5` for the one that was initially
missed. `snippet.py` turned out to have **no direct tests at all** — only
incidental coverage through the formatter, which never touched the
`chunk_pos` window logic vector search relies on to anchor a snippet near
the hit. All of `build_ts_query`, `extract_title`, `chunk_document`,
`reciprocal_rank_fusion` and `reindex_collection` got their listed cases.

`config.py`'s `sqlalchemy_url` separator was missed on the first pass and
caught on re-reading this file — worth recording *why* it hid: coverage.py
doesn't instrument conditional expressions, so
`separator = "&" if "?" in url else "?"` read as 100% covered with its `&`
arm never executed. It matters for any DSN carrying `?sslmode=require`,
where the wrong separator emits a second `?` and only fails later, at
connect time, far from the code.

## Architecture assessment

**Is it solid enough? Yes.** That verdict stands unchanged. The service
facade, content-addressing, per-model embedding tables, schema-qualified
pgvector handling, and one-directional layering were all right as found and
needed no work.

⚠️ **The ACL note was the most important item in this section, and the
easiest to skim past** — `caa8c1a`. The review flagged that `list_contexts`,
`context_check` and `_resolve_owned_collection` filter by `owner_user_id`
directly, so grants will need a resolution change there too, and said "worth
a comment now so future-you doesn't assume the swap is literally one
function." Meanwhile `auth.py` asserted the opposite in its module
docstring. That claim is now corrected and the two call-site shapes
documented — filter-then-check (ready) versus owner-prefiltered (not) —
with each of the three sites marked in place.

### Coupling

1. ✅ **Global engine/settings singletons** — `2416348`, via `reset_engine()`.
2. ✅ **`store.py` at 1076 lines** — `b045970`. Now a package layered
   `_common → cleanup/documents → collection/context/indexing/retrieval`.
   Verified as a pure move by AST comparison: all 67 top-level definitions
   present exactly once, bodies byte-identical. `__init__.py` re-exports the
   public API, so no CLI/MCP/service caller changed. Named `collection.py`,
   not `collections.py`, to avoid shadowing the stdlib.
3. ✅ **`hybrid_query`'s 14 parameters** — `fa286f1`. Frozen `ModelConfig`
   (with `from_settings`) and `QueryOptions`; 14 → 6. Defaults match the old
   per-parameter defaults exactly, so `QueryOptions()` is the old
   all-defaults call.
4. ⏸️ **N+1 patterns** — deliberately left. `list_collections` still runs a
   stats query per collection, and `search_fts`/`search_vec` still call
   `get_context_for_path` per result row. ~40 fast queries per search at a
   20-result limit; batch-fetch when it starts to matter.
5. ⏸️ **Not worth changing** — per-call `LlmClient` construction,
   `SearchResult` living in `fts.py`, CLI/MCP result-shaping duplication.
   Unchanged, still fine.

## Found during the work

Three bugs the new tests surfaced, none visible from reading the code:

- **`collection exclude` inverted its own meaning** — `1ffc0d5`. Excluding
  the only collection made unscoped search return *everything* instead of
  nothing: `_resolve_collection_names` returned `[]`, which meant "no
  filter" to `resolve_collection_ids` (all collections) while
  `_filter_by_collections` no-opped on `len(names) <= 1`. Same shape in
  `mcp/server.py`. All three filters now distinguish "nothing in scope"
  from "one collection, already scoped in SQL".
- **`expand_query` didn't catch `IndexError`** — `1ffc0d5`. `chat_json`
  subscripts `choices[0]`, so a router answering `{"choices": []}` escaped
  the fallback and failed the query instead of degrading.
- **The package split silently weakened the ACL proof** — `b045970`.
  `test_acl_gating.py` proves every choke point gates by patching one name;
  splitting `store.py` spread `can_access` across two modules, and any
  future module importing it directly would have escaped the proof while
  still *looking* gated. `list_collections` now goes through a `can_read`
  helper so there is exactly one import site, and
  `tests/test_acl_import_surface.py` asserts that by AST scan.

## Suggested priority order (all done)

1. ✅ `_expand_braces` recursion + tests — `83ff6c2`
2. ✅ Engine-reset fixture + CLI/MCP coverage — `2416348`
3. ✅ Wire the dead validators — `734d406`
4. ✅ `httpx.MockTransport` tests for `LlmClient` — `6936a4a`
5. ✅ Pure-unit edge cases — `3985a35`, `147b4e5`
6. ✅ `store.py` split and `QueryOptions` — `b045970`, `fa286f1`

Plus, from re-reading this file after the fact: `caa8c1a` (ACL claim),
`7711455` (rerank fallback), `147b4e5` (config branch), and `8c56980`
(the committed coverage snapshot under `devs/`).

## Still open

Nothing from this review. The remaining low-coverage modules are deliberate
and recorded in `devs/COVERAGE.md`: `cli/commands/mcp.py` (31%, daemon
start/stop needing real process spawning), the `bench` CLI wrapper (47%,
while `bench.py` itself is 93%), and `mcp/server.py`'s REST routes, which
only register under `marq mcp --http`.
