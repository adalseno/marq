# Code review — marq (qmd-py), second pass

*Senior-developer review, 2026-07-27, following up on
`code-review-20260727.md` after all of its findings were resolved.
Reviewed at commit `903c083`. Scope: full re-read of the source with fresh
eyes — correctness, robustness, performance, and what the last review's
fixes may have left behind.*

*Resolved 2026-07-28 across eight commits, `3bd549e`…`a2dec9b`, merged to
`master` as a fast-forward. Nine of the eleven findings are closed; the
two left open (7 and 8) are the watchlist items the review itself filed as
deliberate debt with no action requested. This file is kept as the record
of what was found, what was done, and what was measured.*

## Outcome

|                        | Before  | After          |
|------------------------|---------|----------------|
| Line coverage          | 90%     | **91%**        |
| Tests passing          | 377     | **387**        |
| Runnable without infra | 250     | **253** (~1s)  |
| Findings closed        | —       | **9 of 11** (7, 8 deferred by design) |

Measured effects of the three performance findings, against the real
router and the local disposable Postgres rather than estimated:

| Change | Before | After |
|--------|--------|-------|
| `/tokenize`, 40 candidates (4) | 0.74s sequential | **0.13s** gathered — 5.8x, identical output |
| Embedding calls per 3-variant query (5) | 3 requests | **1** request |
| `INSERT`s for a 5-chunk document (6) | 5 statements | **1** executemany |

## TL;DR (original)

The first review's diagnosis and fixes genuinely landed: the suite is 377
tests / 90% coverage and the 250 no-infra tests run in under a second; the
store split, `QueryOptions`, the validators, and the rerank fallback are
all real improvements, not paperwork. The remaining room for improvement
is **small and concentrated**: two live-confirmed bugs in text handling at
the SQL boundary (both one-line-ish fixes), one REST robustness gap, and a
set of latency wins in the query/embed pipelines that are now the biggest
practical lever. Nothing architectural needs to move.

Verification for this pass ran against the local disposable Postgres
(`localhost:5433`) using `tests/fixtures/sample-collection` as a probe
collection, removed afterward.

## Bugs

1. ✅ **A term with a leading apostrophe crashes FTS with a raw traceback**
   — `3bd549e`. Fixed as suggested: `sanitize_fts_term` now strips
   leading *and* trailing apostrophes and all-apostrophe terms sanitize to
   empty, which callers already drop. Note the review's own text says
   trailing ones parse fine (`l'appel:*`) while its Fix line says to strip
   both — stripping both is what landed, since a lexeme can't end with one
   either and `sports'` → `sports` only widens the match. Pinned with the
   three requested tests (`"'"`, `"rock 'n roll"`, `"don't"`) plus a live
   integration test. Reproduced on the pre-fix code before fixing: the
   traceback shows `'to_tsquery_2': "rock:* & 'n:* & roll:*"` in the
   parameters, exactly as described.

   *Original finding:* confirmed live. `sanitize_fts_term`
   (`src/qmd_py/search/fts.py:84`)
   keeps `'` anywhere in a term, but in tsquery input the apostrophe is
   the *lexeme-quote character*: a term that begins with one opens a
   quoted lexeme that never closes. `marq search "rock 'n roll"` builds
   `rock:* & 'n:* & roll:*`, Postgres raises
   `syntax error in tsquery`, and the user gets an unhandled
   `ProgrammingError` traceback. A bare `'` (→ `':*`) does the same.
   Interior/trailing apostrophes are fine (`don't:*`, `l'appel:*` both
   parse).

   Blast radius: `marq search`, `marq query` (both the initial BM25 probe
   and any `lex:` sub-query — `hybrid_query`'s fallback catches only
   httpx/JSON-shape errors, not DB errors), and the MCP `query` tool. The
   typed-query validators can't help: this is a syntactically legal lex
   query; the defect is in sanitization.

   Fix: strip leading/trailing apostrophes in `sanitize_fts_term` (a
   lexeme can't start or end with one anyway) and let all-apostrophe
   terms sanitize to empty, which callers already drop. Pin with tests
   for `"'"`, `"rock 'n roll"`, and `"don't"` (the last to guard the
   still-working interior case).

2. ✅ **`marq ls <collection>/<prefix>` treats the prefix as a LIKE
   pattern** — `1d0aaac`, using the suggested
   `startswith(path_prefix, autoescape=True)`. Verified live both ways:
   on the pre-fix code `marq ls <coll>/doc_` listed `docs/architecture.md`
   (the `_` matching any character); after, `doc_` and `%` match nothing
   while the literal `docs` prefix still works.

   *Original finding:* confirmed live. `list_files`
   (`src/qmd_py/store/retrieval.py:479`)
   does `.like(f"{path_prefix}%")` without escaping, so `_` and `%` in
   the user's path act as wildcards: against the sample fixture,
   `marq ls review-probe/s_c` listed all of `src/`. Underscores in
   directory names are common enough that this will eventually surprise
   someone. Fix:
   `col(Document.path).startswith(path_prefix, autoescape=True)`.

3. ✅ **REST `/query` returns 500 instead of 400 on malformed `searches`
   entries** — `08e8dc7`, wrapped in the suggested
   `try/except (ValidationError, AttributeError, TypeError)`. Tested
   through the real ASGI app with `httpx.ASGITransport`, covering a
   non-dict entry, an invalid `type`, `None`, a non-list `searches`, and
   an absent one. This is also what lifted `mcp/server.py` from 79% to
   84% and narrowed the coverage gap the review's own Coverage section
   describes.

   *Original finding:* (code inspection; the route only exists under
   `marq mcp --http`). `src/qmd_py/mcp/server.py:733` builds
   `SubSearch(type=s.get("type"), ...)` directly from request JSON: a
   non-dict entry (`"searches": ["foo"]`) raises `AttributeError`, and an
   invalid `type` value raises pydantic `ValidationError` — both escape
   the handler and surface as a 500, unlike every other malformed-input
   path in the same function which correctly 400s. Wrap the construction
   in `try/except (ValidationError, AttributeError, TypeError)` → 400.

## Performance (the biggest remaining headroom)

None of these are bugs, and all can wait until latency actually bothers
you — but the query path now does noticeably more sequential I/O than it
needs to.

4. ✅ **Up to 40 sequential `/tokenize` round trips per reranked query** —
   `4340e2d`. `asyncio.gather` as suggested; measured at **5.8x** on the
   tokenize phase (0.74s → 0.13s for 40 candidates against the real
   router) with byte-identical truncation output. Two things checked
   rather than assumed: `gather` propagating the first exception does
   *not* leave the siblings' exceptions unretrieved (CPython's
   `_done_callback` marks them), so the existing rerank fallback stays
   clean; and the 40-way fan-out is well inside httpx's default
   100-connection pool and the client's 120s timeout, which the live run
   confirms.

   *Original finding:*
   `hybrid_query` awaits `_rerank_safe_text` in a loop
   (`src/qmd_py/search/hybrid.py:565`), and any chunk over 800 chars pays
   a `/tokenize` call — chunks run up to 2700 chars, so most candidates
   do. These calls are independent and `httpx.AsyncClient` is
   concurrency-safe: `asyncio.gather` here is the single easiest
   query-latency win in the codebase.

5. ✅ **One embedding HTTP call per vec/hyde sub-query** — `7d31d01`.
   Exactly the suggested shape: `search_vec` gained an optional
   `query_embedding`, and `hybrid_query` embeds every variant in one
   `/v1/embeddings` request, guarded by a `has_embeddings_table()` check
   so the nothing-embedded-yet case still costs zero embedding calls. The
   sub-searches themselves stay sequential, per the review's note about
   the shared `AsyncSession`. Pinned by a test asserting one request of
   size 3 rather than three requests. Doing this exposed a latent test
   defect: both mock routers returned a single fixed vector regardless of
   batch size, which no real router does — they now echo one per input.

   *Original finding:* The sub-searches
   themselves can't naively be gathered (they share one `AsyncSession`),
   but the embedding round trips can be collapsed: `LlmClient.embed`
   already takes a batch, so embedding all vec/hyde query variants in one
   `/v1/embeddings` request — with `search_vec` accepting a precomputed
   vector — removes N−1 round trips per query.

6. ✅ **`embed_pending_documents` re-queries and single-inserts in its
   inner loop, inside one giant transaction** — `04a5fe6`. All three
   parts: `_pgvector_schema` and the INSERT hoisted above the loop, one
   `executemany` per document (verified by SQL logging — a 5-chunk
   document now logs one `INSERT` statement, not five), and a commit per
   document so a long run is restartable, pinned by a test that fails the
   router on the third document and confirms a resumed run has one left.
   One thing the finding didn't anticipate: with per-document commits as
   the only commit points, a run with nothing pending would roll back
   `ensure_embedding_model`'s row *and* its `CREATE TABLE` (DDL is
   transactional in Postgres), so there is now a commit right after that
   registration.

   *Original finding:* `_pgvector_schema(session)` is
   called once per *document* (`src/qmd_py/search/vector.py:249`) though
   the answer never changes — hoist it above the loop. Chunk inserts go
   one row at a time (`executemany` would do). And the CLI commits only
   after the whole run (`cli/commands/write.py:318`), so a crash midway
   through a large embed loses everything even though the pipeline is
   naturally resumable (`hash NOT IN (...)`). Committing per document (or
   per small batch) makes long embed runs restartable and keeps the
   transaction short on the shared server.

## Design notes / watchlist

Deliberate-debt items to keep on the radar, none urgent:

7. ⏸️ **Scoped vector search can miss a small collection entirely** —
   left as filed. Still TS-parity and still fine at current scale; the
   remedy named here (pgvector 0.8 iterative index scans) is the right
   one when multi-collection usage grows.

   *Original finding:* The
   candidate CTE is global by design (only `ORDER BY distance LIMIT n`
   uses the HNSW index), with the collection filter applied outside. With
   `limit*3` global candidates, a `-c small-collection` query in a
   database dominated by other collections can return nothing even though
   the target collection has relevant vectors — all the nearest global
   neighbours are foreign. TS-parity today and fine for the current
   scale; when multi-collection usage grows, pgvector 0.8's iterative
   index scans (`SET hnsw.iterative_scan = relaxed_order`) are the
   DB-side remedy.

8. ⏸️ **`get`/`multi-get`/glob are O(total documents) per lookup** — left
   as filed, for the reason given: fine at thousands of documents, worth
   pushing the exact-path and docid-prefix cases into SQL before tens of
   thousands.

   *Original finding:*
   `_active_document_refs` loads every active document row (minus body)
   into Python for each resolution. Perfectly fine at thousands of
   documents; worth moving the exact-path and docid-prefix cases into SQL
   before the index reaches tens of thousands.

9. ✅ **`marq update` aborts the whole run on the first failing update
   command** — `e2e0444`. Failures are collected and reported together at
   the end, exit code still non-zero. A collection whose update command
   failed is *not* reindexed (its source tree may be half-updated) but
   the ones after it are. Extended slightly beyond the finding: an
   `OSError` from `reindex_collection` (a vanished source directory) is
   handled the same way, since it strands later collections identically.
   The regression test was confirmed to fail against the old code — exit
   3, second collection never reached.

   *Original finding:* (`cli/commands/write.py:271-273`): collections after the
   failing one never reindex. Continuing and reporting failures at the
   end (with a non-zero exit) is friendlier for the cron-style usage this
   command invites.

10. ✅ **`multi_get`'s comma-vs-glob heuristic omits `[`** — `1d0aaac`.
    Took the "one character to fix" option over the "document it" one:
    `[` now joins `*?{`, so `a[1].md,b.md` is treated as one glob like
    every other metacharacter, and `[seq]` still works as a plain glob.
    Pinned both ways, as asked.

    *Original finding:*
    (`store/retrieval.py:365` checks only `*?{`) while `_matches_pattern`
    honours fnmatch's `[seq]` — so `"a[1].md,b.md"` is misclassified as a
    comma list. One character to fix, or one sentence to document
    alongside the existing comma-glob caveat; either way, pin it.

11. ✅ **`llm_base_url` defaults to a personal hostname** — `a2dec9b`.
    Took the "localhost placeholder" option over "make it required":
    requiring it would stop `status`/`ls`/`get`/`search` working for
    someone with no router at all, since only
    `query`/`vsearch`/`embed`/`doctor` need one. Verified on a simulated
    fresh install whose `.env` has only `MARQ_POSTGRES_URL` — `status`
    runs normally and `doctor` reports `⚠ LLM router: unreachable`
    against `localhost:8099`. A follow-up commit scrubbed the hostname
    from the remaining comments (migration, conftest, `compose.yaml`,
    `llm-stack/`) in favour of `yourserver.com`, matching `.env.dev`'s
    existing placeholder, and corrected `llm-stack/README.md`, which
    still claimed the old default.

    *Original finding:*
    `http://ubuserver.internal:8099` is baked into `config.py:32` (and
    echoed in the README table). Anyone else's install fails with a
    confusing connect error to a host that doesn't exist for them instead
    of "not configured". Default to a localhost placeholder or make the
    setting required now that the project is heading to GitHub.

## Coverage

`devs/COVERAGE.md`'s stated gaps (daemon start/stop, the bench click
wrapper, HTTP-only REST routes) held up under reading — they are
genuinely the right things to leave. The only additions this review asks
for are the regression tests attached to findings 1–3 and 10.

*Resolution: all four landed, plus tests for findings 5, 6 and 9 — ten
new tests, 377 → 387. `devs/COVERAGE.md` was refreshed against the new
suite (90% → 91%); the REST route gap it describes is now narrower, since
`/query`'s malformed-input handling is covered even though the success
paths still need a live router.*

## Suggested priority order

1. `sanitize_fts_term` apostrophe fix + tests (finding 1) — user-facing
   crash on legitimate input.
2. `list_files` LIKE autoescape (finding 2) and the REST 400 guard
   (finding 3) — small, mechanical.
3. `asyncio.gather` the `/tokenize` calls (finding 4).
4. The rest as they start to itch.

*Followed in that order. "The rest" turned out to be worth doing in the
same pass: findings 5, 6, 9 and 11 all landed too, leaving only the two
scale-dependent watchlist items (7, 8) open.*

## Verification

Beyond the suite, every claim that pytest alone can't settle was checked
against the local disposable Postgres (`localhost:5433`) and the real
router, using `tests/fixtures/sample-collection` as a probe collection,
removed afterward — `marq status` back to zero collections. Both
live-confirmed bugs (1, 2) were also **reproduced on the pre-fix code**
before fixing, so neither fix chases a theoretical defect. Finding 11 was
checked from a simulated fresh install rather than this machine's
configured one, which is the only place its symptom appears.
