"""CLI tests for the commands that need the LLM router as well as
Postgres: `embed`, `vsearch`, `query`, `doctor`.

Split out from test_cli.py because these are the slow ones - each makes
real embedding/expansion/rerank calls against `MARQ_LLM_BASE_URL`.

Sync on purpose, like test_cli.py: every command body ends in
`asyncio.run()`.
"""

import json
from collections.abc import Callable
from pathlib import Path

import pytest
from click.testing import Result

pytestmark = pytest.mark.integration

Marq = Callable[..., Result]


def _indexed_collection(marq: Marq, tmp_path: Path) -> None:
    (tmp_path / "auth.md").write_text(
        "# Authentication\n\nUsers sign in with an API token. The token is "
        "checked on every request before any handler runs.\n"
    )
    (tmp_path / "cooking.md").write_text(
        "# Sourdough\n\nMix flour and water, rest the dough, then bake it in "
        "a very hot oven.\n"
    )
    result = marq("collection", "add", str(tmp_path), "--name", "docs", "--mask", "**/*.md")
    assert result.exit_code == 0, result.output


@pytest.mark.llm
def test_embed_then_vsearch_finds_semantically_related_document(
    marq: Marq, tmp_path: Path
) -> None:
    _indexed_collection(marq, tmp_path)

    embedded = marq("embed")
    assert embedded.exit_code == 0, embedded.output

    result = marq("vsearch", "how do users log in", "-n", "1")

    assert result.exit_code == 0, result.output
    assert "auth.md" in result.output


def test_vsearch_without_embeddings_reports_no_results(marq: Marq, tmp_path: Path) -> None:
    """No embeddings table yet degrades to an empty result, not a raw
    'relation does not exist' SQL error."""
    _indexed_collection(marq, tmp_path)

    result = marq("vsearch", "anything at all")

    assert result.exit_code == 0, result.output
    assert "No results found." in result.output


@pytest.mark.llm
def test_query_finds_document_via_hybrid_pipeline(marq: Marq, tmp_path: Path) -> None:
    _indexed_collection(marq, tmp_path)
    marq("embed")

    result = marq("query", "how does authentication work", "-n", "1")

    assert result.exit_code == 0, result.output
    assert "auth.md" in result.output


def test_query_no_rerank_skips_the_llm_rerank_pass(marq: Marq, tmp_path: Path) -> None:
    _indexed_collection(marq, tmp_path)
    marq("embed")

    result = marq("query", "authentication token", "--no-rerank", "--explain")

    assert result.exit_code == 0, result.output
    # skip_rerank scores are pure 1/rrf_rank, so the trace reports rerank=0.
    assert "rerank=0.0000" in result.output


def test_query_structured_document_bypasses_expansion(marq: Marq, tmp_path: Path) -> None:
    """A multi-line typed query document skips the expansion LLM call
    entirely - see parse_structured_query."""
    _indexed_collection(marq, tmp_path)

    result = marq("query", "lex: authentication token\nvec: how users sign in", "--no-rerank")

    assert result.exit_code == 0, result.output
    assert "auth.md" in result.output


def test_query_explain_json_includes_score_trace(marq: Marq, tmp_path: Path) -> None:
    _indexed_collection(marq, tmp_path)

    result = marq(
        "query",
        "lex: authentication token\nvec: how users sign in",
        "--no-rerank",
        "--explain",
        "--format",
        "json",
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload
    explain = payload[0]["explain"]
    assert explain["rrf"]["rank"] >= 1
    assert "blendedScore" in explain


def test_query_min_score_filters_matching_results_out(marq: Marq, tmp_path: Path) -> None:
    """Same query that returns auth.md above, but --min-score above any
    achievable score - so the filter, not an empty candidate set, is what
    empties the output."""
    _indexed_collection(marq, tmp_path)
    found = marq("query", "lex: authentication token\nvec: how users sign in", "--no-rerank")
    assert "auth.md" in found.output

    result = marq(
        "query",
        "lex: authentication token\nvec: how users sign in",
        "--no-rerank",
        "--min-score",
        "1.5",
    )

    assert result.exit_code == 0, result.output
    assert "No results found." in result.output


def test_query_rejects_negation_in_a_vec_line(marq: Marq, tmp_path: Path) -> None:
    """Typed sub-queries are validated before any LLM call, so this fails
    fast with an actionable message instead of silently searching for a
    literal '-baseball'."""
    _indexed_collection(marq, tmp_path)

    result = marq("query", "lex: sports\nvec: sports -baseball")

    assert result.exit_code == 1
    assert "vec: Negation (-term) is not supported" in result.output
    assert "Use lex for exclusions" in result.output


def test_query_rejects_unmatched_quote_in_a_lex_line(marq: Marq, tmp_path: Path) -> None:
    _indexed_collection(marq, tmp_path)

    result = marq("query", 'lex: sports "unclosed\nvec: sports')

    assert result.exit_code == 1
    assert "unmatched double quote" in result.output


def test_query_still_allows_negation_in_a_lex_line(marq: Marq, tmp_path: Path) -> None:
    """Regression guard on the wiring: lex negation is the supported way
    to exclude, so validation must not reject it."""
    _indexed_collection(marq, tmp_path)

    result = marq("query", "lex: authentication -sourdough\nvec: signing in", "--no-rerank")

    assert result.exit_code == 0, result.output
    assert "auth.md" in result.output


def test_plain_query_with_a_dash_is_not_validated(marq: Marq, tmp_path: Path) -> None:
    """Single-line queries aren't typed sub-queries: they're auto-expanded,
    so the typed-syntax validators must not fire on them."""
    _indexed_collection(marq, tmp_path)

    result = marq("query", "authentication -sourdough", "--no-rerank")

    assert result.exit_code == 0, result.output


def test_doctor_reports_healthy_postgres_and_router(marq: Marq) -> None:
    result = marq("doctor")

    assert result.exit_code == 0, result.output
    assert "marq doctor" in result.output
    assert "Effective configuration:" in result.output
    assert "MARQ_EMBED_MODEL" in result.output


def test_doctor_redacts_postgres_credentials(marq: Marq) -> None:
    result = marq("doctor")

    assert "MARQ_POSTGRES_URL" in result.output
    config_line = next(
        line for line in result.output.splitlines() if "MARQ_POSTGRES_URL" in line
    )
    assert "@" not in config_line or ":***@" in config_line
