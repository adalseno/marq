# Code review — marq (qmd-py), third pass

*Senior-developer review, 2026-07-28, following up on
`code-review-20260727-2.md` after nine of its eleven findings were
resolved (7 and 8 remain open as deliberate, scale-dependent debt).
Reviewed at commit `ad35f9f`. Scope: full re-read of the source with
fresh eyes — remaining bugs, missed edge cases, robustness of the
long-running surfaces (MCP daemon, `update` under cron) — plus the
requested design section: a logging strategy for production use.*

*Resolved 2026-07-28 across nine commits, `3a51b5c`…`d736abf`, on branch
`review-20260728`. Both bugs and six of the seven robustness findings are
closed, and the logging strategy is implemented through step 3 of its own
rollout order. Findings 7 (stale `initialize` instructions) and the three
carried watchlist items stay open as the scale-dependent debt the review
itself filed them as. This file is kept as the record of what was found,
what was done about it, and what the work turned up that the review did
not anticipate.*

## Outcome

|                        | Before  | After           |
|------------------------|---------|-----------------|
| Line coverage          | 91%     | **92%**         |
| Tests passing          | 387     | **420**         |
| Runnable without infra | 253     | **275** (~10s)  |
| Findings closed        | —       | **8 of 9** (7 deferred by design) |

Per-finding resolution, each closed by the commit named:

| # | Finding | Resolution |
|---|---------|------------|
| 1 | Oversized document aborts indexing | **Fixed** `3a51b5c` — soft-skip at `MAX_INDEXABLE_CHARS` (1,000,000), counted in a new `ReindexResult.skipped_oversize`, reported by both `collection add` and `update`; `update` also catches `SQLAlchemyError` per collection |
| 2 | TOCTOU on `stat()` outside the `try` | **Fixed** `6faf7f7` — stat moved inside; vanished file counts as skipped |
| 3 | Update command can hang forever | **Fixed** `44be9d4` — `stdin=DEVNULL` + 600s timeout, surfaced through the existing per-collection failure list |
| 4 | Suffix match hits mid-filename | **Fixed** `12d1cd5` — match requires a `/` segment boundary |
| 5 | Daemon log grows without bound | **Fixed** `60d7194` — daemon logs through the rotating file handler; raw append retired |
| 6 | One `LlmClient` per MCP query | **Fixed** `6430a8a` — one client for the server's lifetime, closed via a FastMCP lifespan hook |
| 7 | `initialize` instructions frozen | **Open** — deliberate; the `status` tool always returns live numbers |
| 8 | REST accepts JSON booleans as ints | **Fixed** `92bfe29` — `isinstance(x, int) and not isinstance(x, bool)` |
| 9 | HTTP transport has no auth | **Mitigated** `92bfe29` — startup warning on a non-loopback bind; a real token check still belongs with the real ACL |
| — | Logging strategy | **Implemented** `435de08` (step 1), `60d7194` (steps 2–3), `d736abf` (docs) |

### What the work turned up that the review didn't anticipate

- **FastMCP hijacks the root logger.** It calls `logging.basicConfig()`
  with a `RichHandler`, so records propagating up from `qmd_py` were
  emitted twice — once through our handler into the log file, once
  through Rich into the daemon's captured stdio file, in a different
  format. `create_mcp_server()` now sets `propagate = False`. Caught by
  running the daemon and reading both files, not by any test.
- **The request-id filter has to live on the handler, not the logger.**
  A filter attached to a logger does not run for records propagating up
  from its children, and every real call site is a child
  (`qmd_py.search.hybrid`, `qmd_py.store.indexing`, …).
- **`log_duration`'s fields must be set inside the `with` block.** The
  line is emitted on exit, so the first version of the reindex call site
  updated a dict nobody would ever read.
- **The TOCTOU regression test needs `_discover_files` pinned.** That
  function stats every candidate itself (`is_file()`), which drops the
  file before the loop and dodges the very race being simulated.
- **Click's `--help` is eager**, so it exits before the group callback
  that configures logging — the `-v`/`-vv` tests drive a real subcommand
  (`skills list`, which needs no infrastructure) instead.
- **The daemon needs INFO without stealing an explicit `WARNING`.**
  Distinguishing "the user set WARNING" from "WARNING is the default"
  uses pydantic's `model_fields_set`.

### Verification of the fixes

Full suite green (420 passed, including the 145 integration tests),
`ruff check`, `mypy src alembic tests`, and `zensical build --strict`
all clean. Live-verified against the local disposable Postgres and the
real router, with the scratch collection removed afterwards:

- Finding 1's actual failure mode: a 2.9 MB distinct-token file indexed
  alongside good ones — `Indexed: 2 new … Skipped 1 oversized file(s)`
  plus a WARNING naming the file and its size, where the old code raised
  `string is too long for tsvector` and left a committed empty
  collection behind.
- Finding 3: an update command of `read -r line` now fails fast with
  exit 1 instead of blocking forever on inherited stdin.
- Finding 4: `marq get alpha.md` resolves; `marq get pha.md` reports not
  found.
- Logging: `marq mcp` at `MARQ_LOG_LEVEL=DEBUG` answers `initialize`
  with pure JSON-RPC on stdout and every log line on stderr (the
  constraint the whole design hangs on); the HTTP daemon produced
  correlated `WARNING`+`INFO` pairs per request
  (`[rest-3f3ad5] … status=400 2ms`) in the rotating file, with no query
  text and no duplicated lines.

## TL;DR

The codebase is in genuinely good shape: 387 tests / 91% coverage, ruff
and mypy --strict clean, and the second review's fixes all held up under
re-reading. What's left is **one real bug class** (oversized documents
crash indexing with a raw Postgres error — live-confirmed at the SQL
level), a small cluster of **robustness gaps on the unattended paths**
(`marq update` under cron, the MCP daemon), and a handful of watchlist
items. Nothing architectural needs to move. The single biggest missing
production facility is logging, which gets its own section below with a
concrete, incremental implementation plan.

Verification for this pass: the full suite (`uv run pytest`, 387 passed
including integration), `ruff check`, `mypy src alembic tests`, and
targeted read-only SQL probes against the local disposable Postgres
(`localhost:5433`) — pure function calls (`to_tsvector`/`to_tsquery`),
no marq tables touched, nothing to clean up.

## Bugs

1. ✅ **A document over the 1 MiB tsvector limit aborts the whole indexing
   run with a raw Postgres error.** — `3a51b5c`. Live-confirmed at the SQL level:

   ```
   SELECT to_tsvector('english', <1.2 MB of distinct tokens>)
   ERROR:  string is too long for tsvector (1450948 bytes, max 1048575 bytes)
   ```

   `reindex_collection` reads every file matching the pattern with no
   size cap, and `update_document_search_vector`
   (`src/qmd_py/search/fts.py:245`) feeds the whole body to
   `to_tsvector`. The limit is on the *tsvector*, not the raw text —
   repetitive prose dedupes into few lexemes and survives at many MB —
   but a distinct-token-heavy file (a log dump, generated/minified code,
   a data table exported to markdown) crosses it well under 2 MB of
   source text.

   Blast radius is worse than one file:
   - In `marq collection add`, the collection row is committed *before*
     the reindex (`cli/commands/write.py:82-88`), so the failed run
     leaves a committed, empty collection behind — and re-running the
     command says "Collection already exists. Use a different name",
     which is exactly wrong advice.
   - In `marq update`, only `OSError` is caught around
     `reindex_collection` (`cli/commands/write.py:290`). A
     `ProgrammingError` from one oversized file aborts the entire run
     and strands every later collection — the same failure shape the
     second review's finding 9 just fixed, reintroduced through a
     different exception type.

   Fix: skip files whose body exceeds a byte cap at index time and
   report them, the same soft-skip convention `multi_get`'s `max_bytes`
   already established. A default around 1,000,000 characters is the
   honest line (a tsvector can't exceed 1 MiB, and even a *successful*
   giant document degrades search anyway); add a `skipped_oversize`
   bucket to `ReindexResult` so the count is visible rather than
   silent. Optionally also broaden `update`'s per-collection
   `except OSError` to include `SQLAlchemyError` (+ rollback), so no
   future per-file surprise can strand later collections again.

2. ✅ **TOCTOU in `reindex_collection`: the `stat()` call is outside the
   `try`.** — `6faf7f7`. `filepath.read_text()` is guarded, but
   `filepath.stat().st_mtime` two lines later
   (`src/qmd_py/store/indexing.py:192`) is not — a file deleted (or made
   unreadable) between the read and the stat raises an unhandled
   `FileNotFoundError`. Rare, but it's precisely the
   fast-moving-worktree scenario the reindexer is documented to support
   (branch switches over an indexed tree). One-line fix: move the
   `stat()` into the existing `try` (stat before read, or widen the
   block); the file then counts as skipped, which is what the vanished
   file should be.

## Robustness (the unattended paths)

3. ✅ **`marq update`'s update command can hang forever, invisibly.** — `44be9d4`.
   `subprocess.run(..., shell=True, capture_output=True)`
   (`cli/commands/write.py:269`) has no timeout and inherits stdin. A
   `git pull` that decides to prompt for credentials writes the prompt
   into the captured stderr (the user sees nothing) and blocks reading
   stdin — under cron, forever; interactively, a silent hang. This
   command explicitly invites cron usage, so: `stdin=subprocess.DEVNULL`
   (most tools then fail fast instead of prompting), plus a generous
   `timeout=` (e.g. 600s) with `subprocess.TimeoutExpired` handled as an
   ordinary per-collection failure through the existing `failures` list.

4. ✅ **Suffix matching in `_match_named_document` matches mid-filename.** — `12d1cd5`.
   `f"marq://{coll}/{path}".endswith(name)`
   (`src/qmd_py/store/retrieval.py:100`) means `marq get o.py` silently
   returns `src/foo.py` when no file is literally named `o.py`. The
   partial-path feature only intends segment-boundary suffixes
   (`src/foo.py`, `foo.py`); requiring the match to start at a `/`
   boundary (`endswith("/" + name)`) keeps every documented case working
   and kills the surprise. This is a deliberate TS-parity port, so if
   parity wins, document the quirk instead — but it's a one-line
   behavior improvement, and `get`'s "did you mean" suggestions already
   handle the miss gracefully.

5. ✅ **The MCP daemon's log file grows without bound.** — `60d7194`.
   `_start_daemon` appends stdout/stderr to `~/.cache/marq/mcp.log`
   (`cli/commands/mcp.py:61`) forever — no rotation, no cap. For a
   daemon that's meant to run indefinitely this eventually becomes a
   disk problem. Folded into the logging strategy below (a
   `RotatingFileHandler` solves it as a side effect); if logging lands
   later, an interim `logrotate`-style size check at daemon start is
   two lines.

6. ✅ **One `LlmClient` (connection pool) per MCP query call.** — `6430a8a`.
   `_run_query_search` constructs `LlmClient(settings.llm_base_url)`
   inside every call (`src/qmd_py/mcp/server.py:249`), so a long-lived
   server pays TCP/TLS setup and pool warmup per query instead of
   reusing keep-alive connections. The CLI can't do better (one process
   per command), but the server can hold one client for its lifetime —
   `httpx.AsyncClient` is concurrency-safe, which the gathered
   `/tokenize` fan-out already relies on. Modest, easy latency win.

7. ⏸️ **MCP `initialize` instructions are frozen at server construction.** — left open, as filed.
   `create_mcp_server` builds the document counts / "run marq embed"
   hints once (`src/qmd_py/mcp/server.py:801-806`); a daemon running for
   weeks reports stale counts to every newly connecting client. Cheap
   improvement when it itches: rebuild the instructions string lazily
   with a short TTL. Not urgent — the `status` tool always returns live
   numbers.

8. ✅ **REST `/query` accepts JSON booleans where integers are expected.** — `92bfe29`.
   `isinstance(params.get("limit"), int)` is `True` for `true`/`false`
   (Python's `bool` subclasses `int`), so `"limit": true` quietly runs
   with limit 1. Harmless in practice; if the 400-guard ever gets
   another pass, `isinstance(x, int) and not isinstance(x, bool)` is
   the standard fence.

9. ✅ **The HTTP transport has no authentication.** — startup warning in `92bfe29`; the token check still belongs with the real ACL. Fine for the default
   `127.0.0.1` bind and today's single-user reality, but `--host` will
   happily bind `0.0.0.0` while `can_access()` is mocked to allow
   everything — anyone who can reach the port can read every indexed
   document. Until real ACL lands, a one-line warning at startup when
   the bind address is non-loopback would prevent the accidental
   exposure; a bearer-token check belongs in the same future phase as
   the real `can_access()`.

## Watchlist (carried and new)

- ⏸️ **Findings 7 and 8 from the second review stay open** (scoped
  vector search can miss a small collection; `get`/`multi-get`/glob are
  O(total documents)). Still scale-dependent, still fine today.
- ⏸️ **`embed_pending_documents` loads every pending body into memory at
  once** (`src/qmd_py/search/vector.py:241` — the `.all()` includes
  `c.doc`). At thousands of large documents this is a real spike;
  fetching hashes first and bodies per document (or `yield_per`) keeps
  the resumable-commit structure intact. Same size class as review-2's
  finding 8 — note it, don't fix it yet.
- ⏸️ **Embedding-dimension drift has no diagnostic.** If a router preset's
  dimension ever changes behind an already-registered slug,
  `embed` fails with a raw Postgres "expected N dimensions, not M"
  error. `doctor` already talks to both sides; comparing
  `EmbeddingModel.dimension` against a live probe would turn that
  into a `⚠` with an explanation. Low priority, cheap to add.
- ⏸️ **Multi-collection `-c a -c b` filters post-hoc** over a global
  candidate pool (documented in `_filter_by_collections`) — same
  recall caveat as the scoped-vector-search watchlist item, same
  remedy horizon.

## Coverage

91% is a satisfactory plateau, and the remaining gaps are still the
right ones to leave (daemon process management, the bench click wrapper,
REST success paths needing a live router, doctor's failure branches).
Two genuinely missing edge-case pins worth adding regardless of the
fixes above:

- ✅ `d736abf`. A search whose every positive term is a Postgres stopword
  (`marq search "the of"`) — `to_tsquery` reduces it to an empty query
  with a NOTICE (live-confirmed) and matches nothing. That's correct
  behavior, but nothing pins it; a two-line test freezes it.
- ✅ All three landed with their regression tests. The fixes for findings 1–3 each wanted the test
  attached (oversized-file skip counted and reported; vanished-file
  race simulated with a mock `stat`; update-command timeout surfaced as
  a per-collection failure).

## Logging: a strategy that helps without drowning

marq currently has **zero** logging — every degrade path (expansion
fallback, rerank fallback, skipped files, REST 400s) is silent, which is
exactly where a developer chasing "why are my results worse today?"
needs a trace. The design constraints are specific to this project:

**Hard constraint: stdout is never available.** The CLI's stdout is
parseable output (`--format json/csv` pipelines), and the MCP stdio
transport *is* JSON-RPC over stdout — a single stray log line corrupts
the protocol. All logging goes to **stderr and/or a file**, period.

### Shape

One new module, `src/qmd_py/log.py`, ~40 lines:

```python
def setup_logging(level: str, log_file: Path | None = None) -> None:
    root = logging.getLogger("qmd_py")
    root.setLevel(level)
    handler: logging.Handler
    if log_file:
        handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=5_000_000, backupCount=3
        )
    else:
        handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)s %(message)s"
    ))
    root.addHandler(handler)
    # Third-party noise stays quiet unless explicitly asked for:
    for noisy in ("httpx", "httpcore", "sqlalchemy.engine", "mcp", "uvicorn"):
        logging.getLogger(noisy).setLevel(
            logging.DEBUG if level == "DEBUG" else logging.WARNING
        )
```

Every module gets the standard `logger = logging.getLogger(__name__)` —
the `qmd_py.search.hybrid` / `qmd_py.store.indexing` hierarchy comes for
free and lets a developer turn one subsystem to DEBUG later without a
firehose from the rest.

Wire-up points (three, no more):

- `cli/main.py`'s `main()` — before `cli()`, from settings + flags.
- `mcp/server.py`'s `create_mcp_server()` — same call; for the stdio
  transport force the stderr handler (never stdout), for `--daemon` the
  rotating file handler at `~/.cache/marq/marq.log` (which also
  resolves finding 5's unbounded `mcp.log`).
- `tests/conftest.py` — nothing; pytest's `caplog` works out of the box
  with stdlib logging, which is itself a reason to stay on stdlib
  rather than structlog/loguru. This project has no aggregation stack
  to feed; plain formatted lines are grep-able and diff-able, and a
  JSON formatter can be swapped into the same handler later if a
  collector ever appears.

Configuration: `MARQ_LOG_LEVEL` (default `WARNING`) and optional
`MARQ_LOG_FILE` in `Settings` — they follow the existing env/.env
convention and cost two fields. A `-v/--verbose` (INFO) / `-vv` (DEBUG)
eager flag on the click group maps onto the same switch for one-off
debugging without touching the environment.

### What to log at which level — the "useful, not overwhelming" contract

The default (`WARNING`) must be **silent on a healthy run**. That's the
discipline that keeps the log trustworthy: a non-empty log means
something actually degraded.

**WARNING — every silent degrade the code already has.** These are the
highest-value lines in the whole plan, and they're nearly free — each
one is an existing `except` block that today swallows the evidence:

- `expand_query`'s fallback (`search/hybrid.py:178`): *which* exception
  pushed the query down to the no-expansion path.
- The rerank fallback (`search/hybrid.py:590`): same — today a dead
  router and a token-budget bug look identical (silently worse results).
- `reindex_collection`'s skipped files (unreadable, non-UTF-8, and the
  proposed oversize skip): path + reason, once per file.
- `update`'s per-collection failures (already user-visible, but the
  captured stdout/stderr of a failed update command belongs in the log
  even when the console summary truncates it).
- REST 400s and MCP tool-input rejections: the offending payload shape
  (not the full body).

**INFO — one line per unit of work, with timings.** Aimed at the daemon
(`MARQ_LOG_LEVEL=INFO` is the sensible daemon default) and at `-v`:

- Per query: `query done: 3 sub-queries, 40 candidates, reranked,
  9 results, fts=12ms embed=85ms vec=30ms rerank=310ms total=520ms` —
  counts and durations, **never the query text or document content**
  (they're user data; see below). A tiny
  `with log_duration(logger, "rerank"):` context-manager helper keeps
  this from cluttering the pipeline.
- Per reindex: the `ReindexResult` counts plus elapsed time, per
  collection.
- Per embed run: documents/chunks embedded, elapsed, model slug.
- Daemon lifecycle: startup (bind address, schema, model slugs),
  shutdown, per-request method+path+status for the REST routes.

**DEBUG — the full mechanism, content included.** Only here do query
texts, generated tsquery strings, expansion variants, RRF weight
tables, per-candidate rerank scores, and LLM request/response metadata
appear. `sqlalchemy.engine` at INFO (its SQL-echo level) is also gated
behind this. DEBUG is allowed to be overwhelming — that's its job — but
it must remain opt-in per subsystem
(`MARQ_LOG_LEVEL=DEBUG` global, or a targeted
`logging.getLogger("qmd_py.search").setLevel(...)` escape hatch later).

**Privacy note, worth writing into the module docstring:** at WARNING
and INFO, log *shapes* (counts, lengths, paths, durations, exception
types) and never *content* (queries, bodies, snippets). The index holds
whatever the user pointed it at; a log file that quietly re-hosts
fragments of it is a leak vector, especially once the log outlives the
collection (`marq collection remove` deletes documents, not log lines).

**Concurrency:** MCP tool calls are concurrent, so interleaved lines
need correlation. A `contextvars.ContextVar` holding a short random
request id, set at each tool/REST entry point and injected via a
logging `Filter`, makes `%(request_id)s` available in the format string
— ~15 lines, and it's the difference between a usable and a useless
daemon log.

### Rollout order

1. `log.py` + settings + the three wire-up points, with the WARNING
   lines in the existing except/skip blocks — one small PR, immediately
   useful, zero behavior change.
2. INFO timings via the `log_duration` helper in the hybrid pipeline,
   reindex, and embed — the "why is this slow" story.
3. Request-id correlation + daemon file handler (retiring raw
   `mcp.log` appending) — the long-running-server story.
4. DEBUG detail opportunistically, as real debugging sessions reveal
   what's actually missing.

## Suggested priority order

1. Oversized-document skip + `update`'s broadened per-collection catch
   (finding 1) — the only unhandled crash on legitimate input left.
2. Logging step 1 (the WARNING lines cost almost nothing and pay
   immediately).
3. `stat()` into the try (2), `stdin=DEVNULL` + timeout on update
   commands (3) — small, mechanical.
4. Logging steps 2–3, then the rest as they start to itch.

## Verification

- Full suite: `uv run pytest` — **387 passed** (including the 134
  integration tests against the local Postgres and real router);
  `ruff check` and `mypy src alembic tests` both clean at `ad35f9f`.
- Finding 1's failure mode was **reproduced live** as a read-only SQL
  probe against the disposable Postgres (`localhost:5433`):
  ~1.2 MB of distinct tokens → `string is too long for tsvector
  (1450948 bytes, max 1048575 bytes)`. No marq tables were touched and
  no collections were created, so there was nothing to clean up.
- Two adjacent hypotheses were probed the same way and came back
  *fine*, so they are recorded here as non-findings: an
  underscore-only term (`___:*`) and an all-stopword query
  (`the:* & of:*`) both reduce to an empty tsquery with a NOTICE —
  empty results, no error. Worth a pinning test (see Coverage), not a
  fix.
- Findings 2–9 are code-inspection findings with deterministic
  mechanisms (an unguarded `stat()`, an inherited stdin, `endswith`
  semantics, `bool`-is-`int`); none required mutating real
  infrastructure to establish.
