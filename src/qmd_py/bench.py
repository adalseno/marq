"""Search-quality benchmark harness: precision@k/recall/MRR/F1 across
backends (bm25/vector/hybrid/full) against a fixture file - port of the
TS reference's `src/bench/{bench,score,types}.ts`. Framework-agnostic
(no click/CLI dependency here, matching `search/*.py`'s convention) -
progress is reported via an optional `on_progress` callback the CLI
layer supplies.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from sqlalchemy.ext.asyncio import AsyncSession

from qmd_py.auth import CurrentUser
from qmd_py.config import Settings
from qmd_py.llm.client import LlmClient
from qmd_py.search.fts import search_fts
from qmd_py.search.hybrid import (
    ModelConfig,
    QueryOptions,
    hybrid_query,
    parse_structured_query,
)
from qmd_py.search.vector import search_vec

# =============================================================================
# Fixture
# =============================================================================


@dataclass
class BenchmarkQuery:
    """One query in a benchmark fixture, with its expected results.

    Attributes:
        id: Stable identifier, used to label result rows.
        query: The query text. May use the multi-line typed syntax.
        type: Free-form label for grouping (e.g. "lexical", "semantic").
        description: What this query is meant to probe.
        expected_files: Paths that should be retrieved. Matched loosely
            by `paths_match()`, so collection prefixes need not agree.
        expected_in_top_k: The k used for precision@k and hits@k.
    """

    id: str
    query: str
    type: str
    description: str
    expected_files: list[str]
    expected_in_top_k: int


@dataclass
class BenchmarkFixture:
    """A parsed benchmark fixture file.

    Attributes:
        description: What this fixture set covers.
        version: Fixture format version.
        queries: The queries to run.
        collection: Restrict every query to this collection, or None to
            search all of them. A per-file default, overridable at the
            call.
    """

    description: str
    version: int
    queries: list[BenchmarkQuery]
    collection: str | None = None


def load_fixture(path: str | Path) -> BenchmarkFixture:
    """Load and validate a benchmark fixture from JSON.

    Raises:
        ValueError: The file has no `queries` array.
        KeyError: A query is missing a required field.
        json.JSONDecodeError: The file is not valid JSON.
    """
    data = json.loads(Path(path).read_text())
    if not isinstance(data.get("queries"), list):
        raise ValueError("Invalid fixture: missing 'queries' array")
    queries = [
        BenchmarkQuery(
            id=q["id"],
            query=q["query"],
            type=q.get("type", ""),
            description=q.get("description", ""),
            expected_files=q["expected_files"],
            expected_in_top_k=q["expected_in_top_k"],
        )
        for q in data["queries"]
    ]
    return BenchmarkFixture(
        description=data.get("description", ""),
        version=data.get("version", 1),
        queries=queries,
        collection=data.get("collection"),
    )


# =============================================================================
# Scoring - port of score.ts
# =============================================================================


def normalize_path(path: str) -> str:
    """Reduce a path to a comparable form for scoring.

    `marq://collection/docs/readme.md` becomes `docs/readme.md`:
    lowercased, stripped of the scheme, collection prefix and surrounding
    slashes, so a fixture can name files without knowing which
    collection they were indexed into.
    """
    if path.startswith("marq://"):
        without_scheme = path[len("marq://") :]
        slash_idx = without_scheme.find("/")
        path = without_scheme[slash_idx + 1 :] if slash_idx >= 0 else without_scheme
    return path.lower().strip("/")


def paths_match(result: str, expected: str) -> bool:
    """Whether a retrieved path satisfies an expected one.

    Deliberately loose: equal after normalization, or either being a
    suffix of the other. That lets a fixture write `readme.md` for a
    document actually stored at `docs/readme.md` - convenient for
    hand-written fixtures, but it can credit a near-miss, so treat scores
    as comparative rather than absolute.
    """
    nr, ne = normalize_path(result), normalize_path(expected)
    return nr == ne or nr.endswith(ne) or ne.endswith(nr)


def _hits_within(result_files: list[str], expected_files: list[str], k: int) -> int:
    top_k = result_files[:k]
    return sum(1 for expected in expected_files if any(paths_match(r, expected) for r in top_k))


@dataclass
class ScoreMetrics:
    """Retrieval quality for one query against one backend.

    Attributes:
        precision_at_k: Hits in the top k over `min(k, len(expected))`,
            so a query expecting fewer files than k can still reach 1.0.
        recall: Fraction of expected files found anywhere in the results.
        recall_at_1: Recall counting only the top result.
        recall_at_3: Recall counting the top three.
        recall_at_5: Recall counting the top five.
        mrr: Reciprocal rank of the first correct hit; 0.0 if none.
        f1: Harmonic mean of `precision_at_k` and `recall`.
        hits_at_k: Expected files found within the top k.
        matched_files: Expected files that were found, anywhere.
        unmatched_expected_files: Expected files that were missed - the
            useful column when diagnosing a regression.
    """

    precision_at_k: float
    recall: float
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    mrr: float
    f1: float
    hits_at_k: int
    matched_files: list[str]
    unmatched_expected_files: list[str]


def score_results(result_files: list[str], expected_files: list[str], top_k: int) -> ScoreMetrics:
    """Score one ranked result list against its expected files.

    Args:
        result_files: Retrieved paths, best first.
        expected_files: Paths that should have been retrieved.
        top_k: Cutoff for the @k metrics.

    Returns:
        All metrics for this query. An empty `expected_files` yields
        zeros rather than dividing by zero.
    """
    hits_at_k = _hits_within(result_files, expected_files, top_k)
    matched = [e for e in expected_files if any(paths_match(r, e) for r in result_files)]
    matched_set = set(matched)
    unmatched = [e for e in expected_files if e not in matched_set]

    mrr = 0.0
    for i, r in enumerate(result_files):
        if any(paths_match(r, e) for e in expected_files):
            mrr = 1 / (i + 1)
            break

    n_expected = len(expected_files)

    def recall_at(k: int) -> float:
        return _hits_within(result_files, expected_files, k) / n_expected if n_expected else 0.0

    denominator = min(top_k, len(expected_files))
    precision_at_k = hits_at_k / denominator if denominator > 0 else 0.0
    recall = len(matched) / n_expected if n_expected else 0.0
    recall_at_1 = recall_at(1)
    recall_at_3 = recall_at(3)
    recall_at_5 = recall_at(5)
    f1 = (
        2 * (precision_at_k * recall) / (precision_at_k + recall)
        if (precision_at_k + recall) > 0
        else 0.0
    )
    return ScoreMetrics(
        precision_at_k=precision_at_k,
        recall=recall,
        recall_at_1=recall_at_1,
        recall_at_3=recall_at_3,
        recall_at_5=recall_at_5,
        mrr=mrr,
        f1=f1,
        hits_at_k=hits_at_k,
        matched_files=matched,
        unmatched_expected_files=unmatched,
    )


# =============================================================================
# Backends
# =============================================================================


def _unique(files: list[str], limit: int) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for f in files:
        if f in seen:
            continue
        seen.add(f)
        out.append(f)
        if len(out) >= limit:
            break
    return out


class _BackendFn(Protocol):
    async def __call__(
        self,
        session: AsyncSession,
        user: CurrentUser,
        llm_client: LlmClient,
        settings: Settings,
        query: BenchmarkQuery,
        limit: int,
        collection: str | None,
    ) -> list[str]: ...


async def _run_bm25(
    session: AsyncSession,
    user: CurrentUser,
    llm_client: LlmClient,
    settings: Settings,
    query: BenchmarkQuery,
    limit: int,
    collection: str | None,
) -> list[str]:
    parsed = parse_structured_query(query.query)
    if parsed:
        typed, _intent = parsed
        files: list[str] = []
        for q in typed:
            if q.type != "lex":
                continue
            results = await search_fts(
                session, user, q.query, limit=limit, collection_name=collection
            )
            files.extend(r.filepath for r in results)
        return _unique(files, limit)
    results = await search_fts(session, user, query.query, limit=limit, collection_name=collection)
    return [r.filepath for r in results]


async def _run_vector(
    session: AsyncSession,
    user: CurrentUser,
    llm_client: LlmClient,
    settings: Settings,
    query: BenchmarkQuery,
    limit: int,
    collection: str | None,
) -> list[str]:
    parsed = parse_structured_query(query.query)
    if parsed:
        typed, _intent = parsed
        files: list[str] = []
        for q in typed:
            if q.type not in ("vec", "hyde"):
                continue
            results = await search_vec(
                session, user, q.query, llm_client, settings.embed_model,
                limit=limit, collection_name=collection,
            )
            files.extend(r.filepath for r in results)
        return _unique(files, limit)
    results = await search_vec(
        session, user, query.query, llm_client, settings.embed_model,
        limit=limit, collection_name=collection,
    )
    return [r.filepath for r in results]


async def _run_hybrid(
    session: AsyncSession,
    user: CurrentUser,
    llm_client: LlmClient,
    settings: Settings,
    query: BenchmarkQuery,
    limit: int,
    collection: str | None,
    *,
    rerank: bool,
) -> list[str]:
    parsed = parse_structured_query(query.query)
    preexpanded = None
    display_query = query.query
    intent = None
    if parsed:
        typed, intent = parsed
        preexpanded = typed
        display_query = (
            next((q.query for q in typed if q.type == "lex"), None)
            or next((q.query for q in typed if q.type == "vec"), None)
            or query.query
        )
    results = await hybrid_query(
        session,
        user,
        display_query,
        llm_client,
        ModelConfig.from_settings(settings),
        QueryOptions(
            limit=limit,
            collection_name=collection,
            intent=intent,
            skip_rerank=not rerank,
            preexpanded=preexpanded,
        ),
    )
    return [r.file for r in results]


async def _run_hybrid_norerank(*args: Any, **kwargs: Any) -> list[str]:
    return await _run_hybrid(*args, **kwargs, rerank=False)


async def _run_hybrid_rerank(*args: Any, **kwargs: Any) -> list[str]:
    return await _run_hybrid(*args, **kwargs, rerank=True)


BACKENDS: dict[str, _BackendFn] = {
    "bm25": _run_bm25,
    "vector": _run_vector,
    "hybrid": _run_hybrid_norerank,
    "full": _run_hybrid_rerank,
}


# =============================================================================
# Orchestration
# =============================================================================


@dataclass
class BackendResult:
    """One backend's `ScoreMetrics` for one query, plus what it returned.

    Attributes:
        precision_at_k: See `ScoreMetrics`.
        recall: See `ScoreMetrics`.
        recall_at_1: See `ScoreMetrics`.
        recall_at_3: See `ScoreMetrics`.
        recall_at_5: See `ScoreMetrics`.
        mrr: See `ScoreMetrics`.
        f1: See `ScoreMetrics`.
        hits_at_k: See `ScoreMetrics`.
        total_expected: How many files the fixture expected, for reading
            the ratios above in context.
        latency_ms: Wall-clock time for this backend's run. Includes LLM
            round trips, so hybrid backends are not comparable with bm25
            on latency alone.
        top_files: Retrieved paths, best first, truncated for display.
        matched_files: Expected files found.
        unmatched_expected_files: Expected files missed.
    """

    precision_at_k: float
    recall: float
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    mrr: float
    f1: float
    hits_at_k: int
    total_expected: int
    latency_ms: float
    top_files: list[str]
    matched_files: list[str]
    unmatched_expected_files: list[str]


async def _run_query(
    session: AsyncSession,
    user: CurrentUser,
    llm_client: LlmClient,
    settings: Settings,
    backend_fn: _BackendFn,
    query: BenchmarkQuery,
    collection: str | None,
) -> BackendResult:
    limit = max(query.expected_in_top_k, 10)
    start = time.monotonic()
    try:
        result_files = await backend_fn(
            session, user, llm_client, settings, query, limit, collection
        )
    except Exception:  # noqa: BLE001 - unavailable backend (e.g. no embeddings yet) scores 0
        return BackendResult(
            precision_at_k=0,
            recall=0,
            recall_at_1=0,
            recall_at_3=0,
            recall_at_5=0,
            mrr=0,
            f1=0,
            hits_at_k=0,
            total_expected=len(query.expected_files),
            latency_ms=(time.monotonic() - start) * 1000,
            top_files=[],
            matched_files=[],
            unmatched_expected_files=query.expected_files,
        )
    latency_ms = (time.monotonic() - start) * 1000
    scores = score_results(result_files, query.expected_files, query.expected_in_top_k)
    return BackendResult(
        precision_at_k=scores.precision_at_k,
        recall=scores.recall,
        recall_at_1=scores.recall_at_1,
        recall_at_3=scores.recall_at_3,
        recall_at_5=scores.recall_at_5,
        mrr=scores.mrr,
        f1=scores.f1,
        hits_at_k=scores.hits_at_k,
        total_expected=len(query.expected_files),
        latency_ms=latency_ms,
        top_files=result_files[:10],
        matched_files=scores.matched_files,
        unmatched_expected_files=scores.unmatched_expected_files,
    )


@dataclass
class QueryResult:
    """Every backend's outcome for a single fixture query.

    Attributes:
        id: The fixture query's identifier.
        query: Its query text.
        type: Its grouping label.
        backends: Backend name to result - the comparison the benchmark
            exists to produce.
    """

    id: str
    query: str
    type: str
    backends: dict[str, BackendResult]


@dataclass
class BenchmarkResult:
    """A complete benchmark run.

    Attributes:
        timestamp: ISO-8601 UTC time the run finished, for comparing runs.
        fixture: Path of the fixture used.
        results: Per-query results, in fixture order.
        summary: Backend name to averaged metrics across all queries -
            the headline table.
    """

    timestamp: str
    fixture: str
    results: list[QueryResult]
    summary: dict[str, dict[str, float]]


def _compute_summary(results: list[QueryResult]) -> dict[str, dict[str, float]]:
    backend_names = {name for r in results for name in r.backends}
    summary: dict[str, dict[str, float]] = {}
    for name in backend_names:
        rows = [r.backends[name] for r in results if name in r.backends]
        if not rows:
            continue
        count = len(rows)
        summary[name] = {
            "avg_precision": sum(b.precision_at_k for b in rows) / count,
            "avg_recall": sum(b.recall for b in rows) / count,
            "avg_recall_at_1": sum(b.recall_at_1 for b in rows) / count,
            "avg_recall_at_3": sum(b.recall_at_3 for b in rows) / count,
            "avg_recall_at_5": sum(b.recall_at_5 for b in rows) / count,
            "avg_mrr": sum(b.mrr for b in rows) / count,
            "avg_f1": sum(b.f1 for b in rows) / count,
            "avg_latency_ms": sum(b.latency_ms for b in rows) / count,
        }
    return summary


async def run_benchmark(
    session: AsyncSession,
    user: CurrentUser,
    llm_client: LlmClient,
    settings: Settings,
    fixture_path: str | Path,
    *,
    collection: str | None = None,
    backend_names: list[str] | None = None,
    on_progress: Callable[[str, str, float], None] | None = None,
) -> BenchmarkResult:
    """Run every fixture query against every selected backend.

    Framework-agnostic on purpose - no click dependency - so the CLI
    layer supplies progress reporting rather than this owning output.

    Args:
        fixture_path: JSON fixture to load.
        collection: Restrict all queries to one collection, overriding
            the fixture's own setting.
        backend_names: Which backends to run (`bm25`, `vector`,
            `hybrid`, `full`). None runs all of them.
        on_progress: Called as `(query_id, backend_name, latency_ms)`
            after each backend finishes, for incremental output.

    Returns:
        Per-query results plus the averaged summary.

    Note:
        Runs real searches against real infrastructure, LLM calls
        included - a full fixture against the hybrid backends takes
        minutes, not seconds.
    """
    fixture = load_fixture(fixture_path)
    active = backend_names or list(BACKENDS.keys())
    effective_collection = collection or fixture.collection

    results: list[QueryResult] = []
    for query in fixture.queries:
        backend_results: dict[str, BackendResult] = {}
        for name in active:
            backend_result = await _run_query(
                session, user, llm_client, settings, BACKENDS[name], query, effective_collection
            )
            backend_results[name] = backend_result
            if on_progress:
                on_progress(query.id, name, backend_result.latency_ms)
        results.append(
            QueryResult(id=query.id, query=query.query, type=query.type, backends=backend_results)
        )

    summary = _compute_summary(results)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    return BenchmarkResult(
        timestamp=timestamp, fixture=str(fixture_path), results=results, summary=summary
    )


# =============================================================================
# Formatting
# =============================================================================


def bench_result_to_json(result: BenchmarkResult) -> str:
    """Serialize a whole run as indented JSON, for diffing runs."""
    return json.dumps(asdict(result), indent=2)


def format_bench_table(results: list[QueryResult]) -> str:
    """Render per-query, per-backend metrics as a fixed-width table."""
    def pad(s: str, n: int) -> str:
        return s[:n].ljust(n)

    def num(n: float) -> str:
        return f"{n:.2f}".rjust(5)

    lines = [
        f"{pad('Query', 25)} {pad('Backend', 8)} {pad('P@k', 6)} {pad('R@1', 6)} "
        f"{pad('R@3', 6)} {pad('R@5', 6)} {pad('MRR', 6)} {pad('F1', 6)} {pad('ms', 8)}",
        "-" * 88,
    ]
    for r in results:
        for backend, br in r.backends.items():
            lines.append(
                f"{pad(r.id, 25)} {pad(backend, 8)} {num(br.precision_at_k)} {num(br.recall_at_1)} "
                f"{num(br.recall_at_3)} {num(br.recall_at_5)} {num(br.mrr)} {num(br.f1)} "
                f"{str(round(br.latency_ms)).rjust(7)}ms"
            )
        lines.append("")
    return "\n".join(lines)


def format_bench_summary(summary: dict[str, dict[str, float]]) -> str:
    """Render the averaged per-backend summary as a fixed-width table."""
    def pad(s: str, n: int) -> str:
        return s[:n].ljust(n)

    def num(n: float) -> str:
        return f"{n:.3f}".rjust(6)

    lines = []
    for name, s in summary.items():
        lines.append(
            f"  {pad(name, 8)} P@k={num(s['avg_precision'])} R@1={num(s['avg_recall_at_1'])} "
            f"R@3={num(s['avg_recall_at_3'])} R@5={num(s['avg_recall_at_5'])} "
            f"MRR={num(s['avg_mrr'])} F1={num(s['avg_f1'])} Avg={round(s['avg_latency_ms'])}ms"
        )
    return "\n".join(lines)
