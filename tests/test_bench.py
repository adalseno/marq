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
    BenchmarkQuery,
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
    assert normalize_path("qmd://sample/docs/readme.md") == "docs/readme.md"


def test_normalize_path_lowercases_and_strips_slashes() -> None:
    assert normalize_path("/Docs/README.md/") == "docs/readme.md"


def test_paths_match_exact() -> None:
    assert paths_match("docs/readme.md", "docs/readme.md")


def test_paths_match_suffix() -> None:
    assert paths_match("qmd://sample/docs/readme.md", "readme.md")
    assert paths_match("readme.md", "qmd://sample/docs/readme.md")


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
