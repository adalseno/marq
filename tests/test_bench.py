"""Phase 10: bench harness tests - scoring functions (pure unit tests)
plus run_benchmark() end to end against the sample-collection fixture
(real Postgres + real llama.cpp router, restricted to the bm25 backend
to keep this fast and to avoid the reranker's live token-budget
sensitivity already covered by test_hybrid.py).
"""

from collections.abc import AsyncIterator

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from qmd_py.auth import CurrentUser
from qmd_py.bench import (
    BackendResult,
    BenchmarkQuery,
    QueryResult,
    _run_bm25,
    _run_query,
    bench_result_to_json,
    format_bench_summary,
    format_bench_table,
    load_fixture,
    normalize_path,
    paths_match,
    run_benchmark,
    score_results,
)
from qmd_py.config import get_settings
from qmd_py.db.models import Collection
from qmd_py.llm.client import LlmClient


def test_normalize_path_strips_qmd_scheme_and_collection() -> None:
    assert normalize_path("marq://sample/docs/readme.md") == "docs/readme.md"


def test_normalize_path_lowercases_and_strips_slashes() -> None:
    assert normalize_path("/Docs/README.md/") == "docs/readme.md"


def test_paths_match_exact() -> None:
    assert paths_match("docs/readme.md", "docs/readme.md")


def test_paths_match_suffix() -> None:
    assert paths_match("marq://sample/docs/readme.md", "readme.md")
    assert paths_match("readme.md", "marq://sample/docs/readme.md")


def test_paths_match_unrelated_files() -> None:
    assert not paths_match("docs/readme.md", "docs/other.md")


def test_score_results_perfect_match() -> None:
    scores = score_results(["a.md", "b.md"], ["a.md"], top_k=1)
    assert scores.precision_at_k == 1.0
    assert scores.recall == 1.0
    assert scores.mrr == 1.0
    assert scores.f1 == 1.0
    assert scores.matched_files == ["a.md"]
    assert scores.unmatched_expected_files == []


def test_score_results_miss() -> None:
    scores = score_results(["b.md", "c.md"], ["a.md"], top_k=1)
    assert scores.precision_at_k == 0.0
    assert scores.recall == 0.0
    assert scores.mrr == 0.0
    assert scores.matched_files == []
    assert scores.unmatched_expected_files == ["a.md"]


def test_score_results_partial_recall_ranked_lower() -> None:
    # expected file found, but at rank 2 (index 1) -> mrr = 1/2
    scores = score_results(["x.md", "a.md"], ["a.md"], top_k=2)
    assert scores.recall == 1.0
    assert scores.mrr == 0.5


def test_score_results_no_expected_files_is_zeroed_not_divide_by_zero() -> None:
    scores = score_results(["a.md"], [], top_k=5)
    assert scores.precision_at_k == 0.0
    assert scores.recall == 0.0


def test_load_fixture_parses_bench_sample_collection_json() -> None:
    fixture = load_fixture("tests/fixtures/bench-sample-collection.json")
    assert fixture.collection == "sample"
    assert len(fixture.queries) == 6
    assert all(isinstance(q, BenchmarkQuery) for q in fixture.queries)
    ids = {q.id for q in fixture.queries}
    assert "exact-http-api" in ids


def test_load_fixture_rejects_missing_queries_array(tmp_path: object) -> None:
    import json
    from pathlib import Path

    bad = Path(str(tmp_path)) / "bad.json"
    bad.write_text(json.dumps({"description": "no queries key"}))
    with pytest.raises(ValueError, match="queries"):
        load_fixture(bad)


@pytest.fixture
async def llm_client() -> AsyncIterator[LlmClient]:
    client = LlmClient(get_settings().llm_base_url)
    yield client
    await client.aclose()


@pytest.mark.integration
async def test_run_benchmark_bm25_only_against_sample_collection(
    session: AsyncSession, user: CurrentUser, llm_client: LlmClient, sample_collection: Collection
) -> None:
    settings = get_settings()
    result = await run_benchmark(
        session,
        user,
        llm_client,
        settings,
        "tests/fixtures/bench-sample-collection.json",
        collection="sample",
        backend_names=["bm25"],
    )
    assert len(result.results) == 6
    assert all("bm25" in r.backends for r in result.results)
    assert "bm25" in result.summary
    # The exact-keyword query should score perfectly on plain BM25.
    exact = next(r for r in result.results if r.id == "exact-http-api")
    assert exact.backends["bm25"].precision_at_k == 1.0


@pytest.mark.integration
async def test_run_benchmark_vector_and_full_backends_against_sample_collection(
    session: AsyncSession, user: CurrentUser, llm_client: LlmClient, sample_collection: Collection
) -> None:
    from qmd_py.search.vector import embed_pending_documents

    settings = get_settings()
    await embed_pending_documents(session, user, llm_client, settings.embed_model, 1024)
    await session.commit()

    result = await run_benchmark(
        session,
        user,
        llm_client,
        settings,
        "tests/fixtures/bench-sample-collection.json",
        collection="sample",
        backend_names=["vector", "hybrid", "full"],
    )
    assert len(result.results) == 6
    assert all(
        {"vector", "hybrid", "full"} <= set(r.backends) for r in result.results
    )
    assert {"vector", "hybrid", "full"} <= set(result.summary)


@pytest.mark.integration
async def test_run_bm25_structured_query_dedupes_across_lex_lines(
    session: AsyncSession, user: CurrentUser, llm_client: LlmClient, sample_collection: Collection
) -> None:
    settings = get_settings()
    query = BenchmarkQuery(
        id="structured",
        query="lex: due date\nlex: priority",
        type="exact",
        description="",
        expected_files=["CHANGELOG.md"],
        expected_in_top_k=3,
    )
    files = await _run_bm25(
        session, user, llm_client, settings, query, limit=10, collection="sample"
    )
    # No duplicate file appears even though both lex lines can match the same doc.
    assert len(files) == len(set(files))
    assert any(f.endswith("CHANGELOG.md") for f in files)


@pytest.mark.integration
async def test_run_query_backend_exception_scores_zero_not_crash(
    session: AsyncSession, user: CurrentUser, llm_client: LlmClient
) -> None:
    settings = get_settings()
    query = BenchmarkQuery(
        id="q", query="anything", type="exact", description="",
        expected_files=["a.md"], expected_in_top_k=1,
    )

    async def broken_backend(*_args: object, **_kwargs: object) -> list[str]:
        raise RuntimeError("backend unavailable")

    result = await _run_query(session, user, llm_client, settings, broken_backend, query, None)
    assert result.precision_at_k == 0
    assert result.recall == 0
    assert result.top_files == []
    assert result.unmatched_expected_files == ["a.md"]


def test_bench_result_to_json_roundtrips() -> None:
    br = BackendResult(
        precision_at_k=1.0, recall=1.0, recall_at_1=1.0, recall_at_3=1.0, recall_at_5=1.0,
        mrr=1.0, f1=1.0, hits_at_k=1, total_expected=1, latency_ms=12.5,
        top_files=["a.md"], matched_files=["a.md"], unmatched_expected_files=[],
    )
    qr = QueryResult(id="q1", query="test", type="exact", backends={"bm25": br})
    from qmd_py.bench import BenchmarkResult

    result = BenchmarkResult(
        timestamp="20260101T000000", fixture="f.json", results=[qr],
        summary={"bm25": {"avg_precision": 1.0}},
    )
    import json

    parsed = json.loads(bench_result_to_json(result))
    assert parsed["fixture"] == "f.json"
    assert parsed["results"][0]["backends"]["bm25"]["precision_at_k"] == 1.0


def test_format_bench_table_includes_query_and_backend_names() -> None:
    br = BackendResult(
        precision_at_k=1.0, recall=0.5, recall_at_1=1.0, recall_at_3=1.0, recall_at_5=1.0,
        mrr=1.0, f1=0.67, hits_at_k=1, total_expected=2, latency_ms=42.0,
        top_files=["a.md"], matched_files=["a.md"], unmatched_expected_files=["b.md"],
    )
    qr = QueryResult(id="my-query", query="test", type="exact", backends={"bm25": br})
    table = format_bench_table([qr])
    assert "my-query" in table
    assert "bm25" in table
    assert "42ms" in table


def test_format_bench_summary_includes_backend_name_and_metrics() -> None:
    summary = {
        "bm25": {
            "avg_precision": 0.5,
            "avg_recall": 0.5,
            "avg_recall_at_1": 0.5,
            "avg_recall_at_3": 0.5,
            "avg_recall_at_5": 0.5,
            "avg_mrr": 0.5,
            "avg_f1": 0.5,
            "avg_latency_ms": 10.0,
        }
    }
    out = format_bench_summary(summary)
    assert "bm25" in out
    assert "P@k=" in out
    assert "Avg=10ms" in out
