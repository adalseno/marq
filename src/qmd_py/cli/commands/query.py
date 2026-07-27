"""`query` (aliased `deep-search`) - hybrid RRF + query-expansion +
reranking search, the "best results" command - port of the TS reference's
`querySearch` handler (src/cli/qmd.ts). `--explain` only affects the
`cli` and `json` formats, matching the TS reference's own documented
scope ("Include retrieval score traces (query, CLI/--format json)").
"""

import json
from datetime import UTC, datetime

import click
from sqlalchemy.ext.asyncio import AsyncSession

from qmd_py.auth import CurrentUser
from qmd_py.cli.commands.read import (
    _echo_empty_search_results,
    _format_options,
    _resolve_collection_names,
    _resolve_format,
)
from qmd_py.cli.formatter import FormatOptions, format_search_results
from qmd_py.cli.runtime import run
from qmd_py.cli.snippet import extract_snippet
from qmd_py.config import get_settings
from qmd_py.llm.client import LlmClient
from qmd_py.search.fts import SearchResult
from qmd_py.search.hybrid import (
    RERANK_CANDIDATE_LIMIT,
    HybridQueryResult,
    ModelConfig,
    QueryOptions,
    hybrid_query,
    parse_structured_query,
    validate_typed_queries,
)
from qmd_py.store import add_line_numbers

_EPOCH = datetime.fromtimestamp(0, tz=UTC)


def _to_search_result(row: HybridQueryResult) -> SearchResult:
    """Adapts a HybridQueryResult onto the SearchResult shape so the
    shared formatters (json/csv/md/xml/toon/files) can be reused as-is -
    the fields those formatters actually read (docid, score, display_path,
    title, context, body, chunk_pos) all exist on both; hash/
    collection_name/modified_at are formatter-irrelevant filler."""
    return SearchResult(
        filepath=row.file,
        display_path=row.display_path,
        title=row.title,
        hash="",
        docid=row.docid,
        collection_name="",
        modified_at=_EPOCH,
        body_length=len(row.body),
        body=row.body,
        context=row.context,
        score=row.score,
        source="hybrid",
        chunk_pos=row.best_chunk_pos,
    )


def _filter_hybrid_by_collections(
    results: list[HybridQueryResult], names: list[str]
) -> list[HybridQueryResult]:
    """Empty name list means nothing is in scope, not "no filter" - see
    read.py's `_filter_by_collections` for the full reasoning."""
    if not names:
        return []
    if len(names) == 1:
        return results
    prefixes = tuple(f"marq://{n}/" for n in names)
    return [r for r in results if r.file.startswith(prefixes)]


def _hybrid_results_to_json(results: list[HybridQueryResult], opts: FormatOptions) -> str:
    """Same shape as `search_results_to_json`, plus an `explain` key per
    row when present - kept separate from the shared formatter since
    `explain` has no equivalent on plain `SearchResult`."""
    entries: list[dict[str, object]] = []
    for row in results:
        snippet_info = (
            extract_snippet(row.body, opts.query, 300, row.best_chunk_pos, None, opts.intent)
            if row.body
            else None
        )
        body = row.body if opts.full else None
        snippet = snippet_info.snippet if (snippet_info and not opts.full) else None
        if opts.line_numbers:
            if body:
                body = add_line_numbers(body)
            if snippet:
                snippet = add_line_numbers(snippet)

        entry: dict[str, object] = {
            "docid": f"#{row.docid}",
            "score": round(row.score, 2),
            "file": row.display_path,
        }
        if snippet_info:
            entry["line"] = snippet_info.line
        entry["title"] = row.title
        if row.context:
            entry["context"] = row.context
        if body:
            entry["body"] = body
        if snippet:
            entry["snippet"] = snippet
        if row.explain:
            entry["explain"] = {
                "rrf": {
                    "rank": row.explain.rrf.rank,
                    "weight": row.explain.rrf.weight,
                    "score": round(row.explain.rrf.score, 4),
                },
                "rerankScore": round(row.explain.rerank_score, 4),
                "blendedScore": round(row.explain.blended_score, 4),
            }
        entries.append(entry)
    return json.dumps(entries, indent=2)


def _echo_explain_summary(results: list[HybridQueryResult]) -> None:
    click.echo()
    for row in results:
        if not row.explain:
            continue
        e = row.explain
        click.echo(
            f"[{row.display_path}] rrf(rank={e.rrf.rank}, weight={e.rrf.weight:.2f}, "
            f"score={e.rrf.score:.4f})  rerank={e.rerank_score:.4f}  blended={e.blended_score:.4f}"
        )


@click.command("query")
@click.argument("query")
@click.option("-c", "--collection", "collections", multiple=True, help="Restrict to collection(s)")
@click.option("-n", "limit", type=int, default=10, show_default=True, help="Number of results")
@click.option("--all", "show_all", is_flag=True, help="Return all matches")
@click.option("--min-score", type=float, default=0.0, help="Minimum score threshold")
@click.option("--full", is_flag=True, help="Show full document content")
@click.option("--intent", default=None, help="Describe what you're after to sharpen ranking")
@click.option(
    "--no-rerank", "skip_rerank", is_flag=True, help="Skip LLM reranking, use RRF scores only"
)
@click.option(
    "-C",
    "--candidate-limit",
    type=int,
    default=RERANK_CANDIDATE_LIMIT,
    show_default=True,
    help="Max RRF candidates to rerank",
)
@click.option("--explain", is_flag=True, help="Include retrieval score traces (cli/json formats)")
@_format_options
def query_command(
    query: str,
    collections: tuple[str, ...],
    limit: int,
    show_all: bool,
    min_score: float,
    full: bool,
    intent: str | None,
    skip_rerank: bool,
    candidate_limit: int,
    explain: bool,
    out_format: str,
    json_flag: bool,
    csv_flag: bool,
    md_flag: bool,
    xml_flag: bool,
    files_flag: bool,
) -> None:
    """Hybrid search: query expansion + BM25/vector RRF fusion + LLM
    reranking (recommended). Accepts a multi-line query document of
    `lex:`/`vec:`/`hyde:` (+ optional `intent:`) lines to bypass automatic
    expansion."""
    out_format = _resolve_format(out_format, json_flag, csv_flag, md_flag, xml_flag, files_flag)
    run(
        _query_impl,
        query,
        collections=collections,
        limit=limit,
        show_all=show_all,
        min_score=min_score,
        full=full,
        intent=intent,
        skip_rerank=skip_rerank,
        candidate_limit=candidate_limit,
        explain=explain,
        out_format=out_format,
    )


async def _query_impl(
    session: AsyncSession,
    user: CurrentUser,
    query: str,
    *,
    collections: tuple[str, ...],
    limit: int,
    show_all: bool,
    min_score: float,
    full: bool,
    intent: str | None,
    skip_rerank: bool,
    candidate_limit: int,
    explain: bool,
    out_format: str,
) -> None:
    names = await _resolve_collection_names(session, user, collections)
    single = names[0] if len(names) == 1 else None
    fetch_limit = 100_000 if show_all else max(50, limit * 2)
    settings = get_settings()

    parsed = parse_structured_query(query)
    preexpanded = None
    effective_intent = intent
    display_query = query
    if parsed:
        typed, parsed_intent = parsed
        error = validate_typed_queries(typed)
        if error is not None:
            click.echo(f"marq: {error}", err=True)
            raise SystemExit(1)
        preexpanded = typed
        effective_intent = intent or parsed_intent
        display_query = next(
            (q.query for q in typed if q.type == "lex"),
            next((q.query for q in typed if q.type == "vec"), query),
        )

    async with LlmClient(settings.llm_base_url) as llm_client:
        results = await hybrid_query(
            session,
            user,
            display_query,
            llm_client,
            ModelConfig.from_settings(settings),
            QueryOptions(
                limit=fetch_limit,
                min_score=min_score,
                candidate_limit=candidate_limit,
                collection_name=single,
                intent=effective_intent,
                skip_rerank=skip_rerank,
                explain=explain,
                preexpanded=preexpanded,
            ),
        )

    results = _filter_hybrid_by_collections(results, names)
    if not show_all:
        results = results[:limit]

    if not results:
        _echo_empty_search_results(out_format)
        return

    opts = FormatOptions(
        full=full, query=display_query, line_numbers=False, intent=effective_intent
    )
    if out_format == "json" and explain:
        click.echo(_hybrid_results_to_json(results, opts))
        return

    click.echo(format_search_results([_to_search_result(r) for r in results], out_format, opts))
    if explain and out_format == "cli":
        _echo_explain_summary(results)
