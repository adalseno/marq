"""Phase 8: hybrid query tests - RRF fusion + structured-query parsing
(pure unit tests) plus expand_query/hybrid_query (integration, against a
real scratch Postgres schema, the real llama.cpp router, and the frozen
tests/fixtures/sample-collection - see conftest.py's `sample_collection`).
"""

from collections.abc import AsyncIterator

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from qmd_py.auth import CurrentUser
from qmd_py.config import get_settings
from qmd_py.db.models import Collection
from qmd_py.llm.client import LlmClient
from qmd_py.search.hybrid import (
    ExpandedQuery,
    RankedResult,
    expand_query,
    hybrid_query,
    parse_structured_query,
    reciprocal_rank_fusion,
)
from qmd_py.search.vector import embed_pending_documents

EMBED_MODEL = "bge-m3-q8_0"
EMBED_DIM = 1024
GENERATE_MODEL = "qwen2.5-3b-instruct-q4_k_m"
RERANK_MODEL = "qwen3-reranker-0.6b-q8_0"


# =============================================================================
# reciprocal_rank_fusion (pure unit tests)
# =============================================================================


def _ranked(file: str, score: float = 0.0) -> RankedResult:
    return RankedResult(file=file, display_path=file, title=file, body="body", score=score)


def test_rrf_sums_contributions_across_lists() -> None:
    list_a = [_ranked("a"), _ranked("b")]
    list_b = [_ranked("b"), _ranked("a")]
    fused = reciprocal_rank_fusion([list_a, list_b])
    # "a" is rank 0 in list_a and rank 1 in list_b, "b" the reverse -
    # symmetric input, so both should tie post-fusion.
    scores = {r.file: r.score for r in fused}
    assert scores["a"] == pytest.approx(scores["b"])


def test_rrf_top_rank_bonus_rewards_rank_zero() -> None:
    # "a" is rank 0 in its only list (+0.05 bonus); "b" is rank 2 in its
    # only list (+0.02 bonus) - "a" should end up strictly ahead.
    fused = reciprocal_rank_fusion([[_ranked("a")], [_ranked("x"), _ranked("y"), _ranked("b")]])
    scores = {r.file: r.score for r in fused}
    assert scores["a"] > scores["b"]


def test_rrf_weights_scale_contribution() -> None:
    fused = reciprocal_rank_fusion([[_ranked("a")], [_ranked("b")]], weights=[2.0, 1.0])
    scores = {r.file: r.score for r in fused}
    # Both rank 0 in their own single-item list (so both get the +0.05
    # top-rank bonus); only the base 1/(k+1) contribution is weighted.
    assert (scores["a"] - 0.05) == pytest.approx(2 * (scores["b"] - 0.05))


def test_rrf_dedupes_same_file_across_lists() -> None:
    fused = reciprocal_rank_fusion([[_ranked("a")], [_ranked("a")]])
    assert len(fused) == 1


def test_rrf_empty_input_returns_empty() -> None:
    assert reciprocal_rank_fusion([]) == []


# =============================================================================
# parse_structured_query (pure unit tests)
# =============================================================================


def test_parse_structured_query_single_line_is_not_structured() -> None:
    assert parse_structured_query("plain query, no newlines") is None


def test_parse_structured_query_parses_typed_lines_and_intent() -> None:
    result = parse_structured_query("lex: foo bar\nvec: semantic foo\nintent: disambiguate")
    assert result is not None
    typed, intent = result
    assert [(q.type, q.query) for q in typed] == [
        ("lex", "foo bar"),
        ("vec", "semantic foo"),
    ]
    assert intent == "disambiguate"


def test_parse_structured_query_is_case_insensitive() -> None:
    result = parse_structured_query("LEX: foo\nVEC: bar")
    assert result is not None
    typed, _ = result
    assert [q.type for q in typed] == ["lex", "vec"]


def test_parse_structured_query_no_intent_line_is_fine() -> None:
    result = parse_structured_query("lex: foo\nhyde: a hypothetical answer")
    assert result is not None
    typed, intent = result
    assert intent is None
    assert len(typed) == 2


def test_parse_structured_query_unprefixed_line_degrades_to_none() -> None:
    assert parse_structured_query("lex: foo\nsome unprefixed line") is None


def test_parse_structured_query_no_typed_lines_degrades_to_none() -> None:
    assert parse_structured_query("intent: only an intent\nintent: again") is None


# =============================================================================
# expand_query (integration - real router)
# =============================================================================


@pytest.fixture
async def llm_client() -> AsyncIterator[LlmClient]:
    client = LlmClient(get_settings().llm_base_url)
    yield client
    await client.aclose()


@pytest.mark.integration
async def test_expand_query_returns_typed_variants(llm_client: LlmClient) -> None:
    expanded = await expand_query(llm_client, "how are tasks stored", GENERATE_MODEL)
    assert expanded
    assert all(isinstance(e, ExpandedQuery) for e in expanded)
    assert all(e.type in ("lex", "vec", "hyde") for e in expanded)


@pytest.mark.integration
async def test_expand_query_falls_back_on_llm_error(llm_client: LlmClient) -> None:
    # An unknown model id makes the router 500 - expand_query must
    # degrade to the fallback pair rather than raising.
    expanded = await expand_query(llm_client, "some query", "not-a-real-model")
    assert [e.type for e in expanded] == ["lex", "vec"]
    assert all(e.query == "some query" for e in expanded)


# =============================================================================
# hybrid_query (integration - real Postgres + router, sample-collection)
# =============================================================================


@pytest.mark.integration
async def test_hybrid_query_finds_relevant_doc(
    session: AsyncSession, user: CurrentUser, llm_client: LlmClient, sample_collection: Collection
) -> None:
    await embed_pending_documents(session, user, llm_client, EMBED_MODEL, EMBED_DIM)
    await session.commit()

    results = await hybrid_query(
        session,
        user,
        "how are tasks stored",
        llm_client,
        EMBED_MODEL,
        GENERATE_MODEL,
        RERANK_MODEL,
        collection_name="sample",
        intent="understand task persistence",
    )
    assert results
    assert any(r.display_path.endswith("tasks.py") for r in results[:3])
    assert all(r.docid for r in results)


@pytest.mark.integration
async def test_hybrid_query_no_rerank_uses_rrf_position_score(
    session: AsyncSession, user: CurrentUser, llm_client: LlmClient, sample_collection: Collection
) -> None:
    results = await hybrid_query(
        session,
        user,
        "priority levels",
        llm_client,
        EMBED_MODEL,
        GENERATE_MODEL,
        RERANK_MODEL,
        collection_name="sample",
        skip_rerank=True,
    )
    assert results
    # 1/rrf_rank scores: strictly descending, first is exactly 1.0.
    assert results[0].score == pytest.approx(1.0)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.integration
async def test_hybrid_query_explain_populates_trace(
    session: AsyncSession, user: CurrentUser, llm_client: LlmClient, sample_collection: Collection
) -> None:
    results = await hybrid_query(
        session,
        user,
        "priority levels",
        llm_client,
        EMBED_MODEL,
        GENERATE_MODEL,
        RERANK_MODEL,
        collection_name="sample",
        explain=True,
        limit=3,
    )
    assert results
    assert all(r.explain is not None for r in results)


@pytest.mark.integration
async def test_hybrid_query_preexpanded_skips_automatic_expansion(
    session: AsyncSession, user: CurrentUser, llm_client: LlmClient, sample_collection: Collection
) -> None:
    """Structured lex:/vec: syntax bypasses expand_query() entirely - a
    deliberately bogus generate_model would blow up expand_query() if it
    were called, so this also proves preexpanded truly skips it."""
    results = await hybrid_query(
        session,
        user,
        "due date migration",
        llm_client,
        EMBED_MODEL,
        "not-a-real-model",
        RERANK_MODEL,
        collection_name="sample",
        preexpanded=[
            ExpandedQuery("lex", "due date migration"),
            ExpandedQuery("vec", "schema changes over time"),
        ],
    )
    assert results
    assert any("CHANGELOG" in r.display_path for r in results)


@pytest.mark.integration
async def test_hybrid_query_min_score_filters_results(
    session: AsyncSession, user: CurrentUser, llm_client: LlmClient, sample_collection: Collection
) -> None:
    results = await hybrid_query(
        session,
        user,
        "priority levels",
        llm_client,
        EMBED_MODEL,
        GENERATE_MODEL,
        RERANK_MODEL,
        collection_name="sample",
        skip_rerank=True,
        min_score=0.99,
    )
    assert all(r.score >= 0.99 for r in results)
