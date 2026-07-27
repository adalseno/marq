"""Hybrid query: query expansion + FTS/vector search per expanded
sub-query + reciprocal rank fusion + chunk-level reranking - port of the
TS reference's hybridQuery/reciprocalRankFusion/expandQuery/rerank
(src/store.ts).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Any

import httpx
from sqlalchemy.ext.asyncio import AsyncSession

from qmd_py.auth import CurrentUser
from qmd_py.cli.snippet import extract_intent_terms
from qmd_py.config import Settings
from qmd_py.llm.client import LlmClient
from qmd_py.search.fts import search_fts, validate_lex_query, validate_semantic_query
from qmd_py.search.vector import chunk_document, search_vec

# Ported constants (src/store.ts) - a strong BM25 signal skips the
# (comparatively expensive) query-expansion LLM call entirely.
STRONG_SIGNAL_MIN_SCORE = 0.85
STRONG_SIGNAL_MIN_GAP = 0.15
RERANK_CANDIDATE_LIMIT = 40

# The router's /rerank batch-size cap is enforced per query+document pair,
# not just per request - measured live at 512 tokens. chunk_document's
# char-based sizing (~3 chars/token) undercounts dense code, so a chunk
# under its own max_chars threshold can still tokenize past 512 and break
# rerank with a raw 500. 400 leaves headroom for the query side of the
# pair; texts shorter than _RERANK_SAFE_CHAR_SKIP skip the extra
# /tokenize round-trip entirely (even 1 char/token can't hit budget).
_RERANK_TOKEN_BUDGET = 400
_RERANK_SAFE_CHAR_SKIP = _RERANK_TOKEN_BUDGET * 2


async def _rerank_safe_text(llm_client: LlmClient, text: str, model: str) -> str:
    """Truncates `text` to fit the router's per-pair rerank token budget,
    measuring with the model's own tokenizer rather than guessing another
    fixed chars/token ratio - that guess is exactly what broke the first
    time (see module-level comment above)."""
    if len(text) <= _RERANK_SAFE_CHAR_SKIP:
        return text
    tokens = await llm_client.tokenize(text, model)
    if len(tokens) <= _RERANK_TOKEN_BUDGET:
        return text
    ratio = len(text) / len(tokens)
    return text[: max(1, int(_RERANK_TOKEN_BUDGET * ratio))]

# =============================================================================
# Query expansion
# =============================================================================


@dataclass
class ExpandedQuery:
    """One typed sub-query, from expansion or spelled out by the caller.

    Attributes:
        type: `"lex"` (BM25 keywords), `"vec"` (semantic) or `"hyde"` (a
            hypothetical answer passage, embedded and searched as though
            it were a document).
        query: The text to search with.
    """

    type: str  # "lex" | "vec" | "hyde"
    query: str


_TYPED_LINE_RE = re.compile(r"^(lex|vec|hyde):\s*(.+)$", re.IGNORECASE)
_INTENT_LINE_RE = re.compile(r"^intent:\s*(.+)$", re.IGNORECASE)


def parse_structured_query(query: str) -> tuple[list[ExpandedQuery], str | None] | None:
    """Multi-line `lex:`/`vec:`/`hyde:` (+ optional `intent:`) query
    syntax - port of the TS reference's `parseStructuredQuery`, minus its
    strict error-throwing for malformed input: an unprefixed line among
    several, or no typed lines at all, just falls through to `None`
    (treated as an ordinary single query, auto-expanded) rather than
    raising - a softer degrade that seemed more appropriate for a CLI
    argument than a hard parse error.
    """
    lines = [line.strip() for line in query.split("\n") if line.strip()]
    if len(lines) <= 1:
        return None

    typed: list[ExpandedQuery] = []
    intent: str | None = None
    for line in lines:
        type_match = _TYPED_LINE_RE.match(line)
        if type_match:
            typed.append(ExpandedQuery(type_match.group(1).lower(), type_match.group(2).strip()))
            continue
        intent_match = _INTENT_LINE_RE.match(line)
        if intent_match:
            intent = intent_match.group(1).strip()
            continue
        return None

    return (typed, intent) if typed else None


def validate_typed_queries(queries: list[ExpandedQuery]) -> str | None:
    """First problem found among explicitly typed sub-queries, or None.

    Deliberately applied only to sub-queries the *caller* spelled out -
    the `lex:`/`vec:`/`hyde:` document syntax and the MCP `query` tool's
    `searches` - never to `expand_query()`'s LLM-generated variants: a
    stray `-term` in those is the model's doing, not a user mistake worth
    failing the search over.
    """
    for q in queries:
        error = (
            validate_lex_query(q.query)
            if q.type == "lex"
            else validate_semantic_query(q.query)
        )
        if error is not None:
            return f"{q.type}: {error}"
    return None


_EXPAND_SYSTEM_PROMPT = (
    "You expand search queries for a hybrid search engine (lexical BM25, "
    "semantic vector, and hypothetical-document search). Given a query and "
    "an optional intent, produce one lexical rephrasing (lex), one semantic "
    "rephrasing (vec), and one short hypothetical answer passage (hyde)."
)

_EXPAND_JSON_SCHEMA = {
    "name": "expanded_query",
    "schema": {
        "type": "object",
        "properties": {
            "lex": {"type": "string"},
            "vec": {"type": "string"},
            "hyde": {"type": "string"},
        },
        "required": ["lex", "vec", "hyde"],
    },
}


async def expand_query(
    llm_client: LlmClient, query: str, model: str, intent: str | None = None
) -> list[ExpandedQuery]:
    """Typed query variants (lex/vec/hyde) for RRF fusion - port of the TS
    reference's `expandQuery()`.

    Uses JSON-schema-constrained chat completion rather than the TS
    reference's local node-llama-cpp GBNF-grammar-constrained generation.
    llama.cpp's server *does* accept a raw `grammar` field over HTTP, but
    that specific line-grammar (`type ": " content "\\n"`) proved
    unreliable in manual testing against this project's own
    qwen2.5-3b-instruct-q4_k_m preset: the model satisfied the grammar's
    syntax while ignoring its semantics (emitting dozens of "lex: " lines,
    never "vec"/"hyde"). JSON-schema output was reliable in the same
    testing and is the more portable mechanism for a pure-HTTP client
    generally, so that's what qmd-py builds on instead.
    """
    user_content = f"Expand this search query: {query}"
    if intent:
        user_content += f"\nQuery intent: {intent}"

    try:
        data = await llm_client.chat_json(
            messages=[
                {"role": "system", "content": _EXPAND_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            model=model,
            json_schema=_EXPAND_JSON_SCHEMA,
        )
    except (httpx.HTTPError, IndexError, KeyError, ValueError, TypeError):
        # IndexError included because chat_json subscripts choices[0]: a
        # router answering {"choices": []} is exactly the misbehavior this
        # fallback exists for, and it used to escape and fail the search.
        return [ExpandedQuery("lex", query), ExpandedQuery("vec", query)]

    expanded = []
    for type_ in ("lex", "vec", "hyde"):
        text = str(data.get(type_, "")).strip()
        if text and text != query:
            expanded.append(ExpandedQuery(type_, text))
    return expanded or [ExpandedQuery("lex", query), ExpandedQuery("vec", query)]


# =============================================================================
# Reciprocal rank fusion
# =============================================================================


@dataclass
class RankedResult:
    """A result reduced to what fusion needs.

    Deliberately narrower than `SearchResult`: RRF only needs identity
    and position, so lexical and vector hits collapse to one shape before
    being fused. `file` is the identity key that matches them up.

    Attributes:
        file: Virtual URI - the key results are deduplicated on.
        display_path: `<collection>/<path>`.
        title: Extracted heading or filename stem.
        body: Full document text.
        score: Fused RRF score, not the original engine's score.
    """

    file: str
    display_path: str
    title: str
    body: str
    score: float


@dataclass
class _RrfEntry:
    result: RankedResult
    rrf_score: float
    top_rank: int


def reciprocal_rank_fusion(
    result_lists: list[list[RankedResult]], weights: list[float] | None = None, k: int = 60
) -> list[RankedResult]:
    """Fuse several ranked lists into one by reciprocal rank.

    Each list contributes `weight / (k + rank + 1)` per document, summed
    across lists, so appearing in several lists beats ranking highly in
    one. Scores from the underlying engines are ignored entirely - only
    positions matter, which is what makes lexical and vector results
    comparable at all.

    Args:
        result_lists: Ranked lists, best first. Empty lists are harmless.
        weights: Per-list multipliers, positional. Lists beyond the end
            default to 1.0, so a short list is not an error.
        k: Damping constant. Larger flattens the difference between
            ranks; the default of 60 is the value from the original RRF
            paper.

    Returns:
        One entry per distinct `file`, ordered by descending fused score.
        Documents ranked first in any list get a small bonus, and ones in
        the top three a smaller one, to break ties toward strong single
        signals.
    """
    weights = weights or []
    entries: dict[str, _RrfEntry] = {}

    for list_idx, result_list in enumerate(result_lists):
        weight = weights[list_idx] if list_idx < len(weights) else 1.0
        for rank, result in enumerate(result_list):
            contribution = weight / (k + rank + 1)
            entry = entries.get(result.file)
            if entry is not None:
                entry.rrf_score += contribution
                entry.top_rank = min(entry.top_rank, rank)
            else:
                entries[result.file] = _RrfEntry(result, contribution, rank)

    fused: list[RankedResult] = []
    for entry in entries.values():
        score = entry.rrf_score
        if entry.top_rank == 0:
            score += 0.05
        elif entry.top_rank <= 2:
            score += 0.02
        fused.append(replace(entry.result, score=score))

    fused.sort(key=lambda r: r.score, reverse=True)
    return fused


def _hybrid_rrf_weights(query_types: list[str]) -> list[float]:
    """Original-query retrieval paths (the primary evidence) get 2x
    weight; expansion-derived (lex/vec/hyde) lists stay at 1x regardless
    of insertion order."""
    return [2.0 if t == "original" else 1.0 for t in query_types]


# =============================================================================
# Hybrid query orchestration
# =============================================================================


@dataclass
class RrfExplain:
    """The fusion half of a `--explain` trace.

    Attributes:
        rank: Position after fusion, 1-based.
        weight: How much the fused position counted toward the blend.
            1.0 means the rerank pass did not run - either `--no-rerank`,
            or the router failed and the query degraded to RRF ordering.
        score: `1 / rank`, the positional component of the blend.
    """

    rank: int
    weight: float
    score: float


@dataclass
class HybridQueryExplain:
    """Deliberately simpler than the TS reference's HybridQueryExplain/
    RRFContributionTrace (no per-list contribution breakdown) - the
    numbers that matter for understanding *why* a result ranked where it
    did (RRF rank/weight, rerank score, blended score) are all here."""

    rrf: RrfExplain
    rerank_score: float
    blended_score: float


@dataclass
class HybridQueryResult:
    """One result from the full hybrid pipeline.

    Attributes:
        file: Virtual URI, `marq://<collection>/<path>`.
        display_path: `<collection>/<path>`.
        title: Extracted heading or filename stem.
        body: Full document text.
        best_chunk: The chunk that scored best against the query - what
            the reranker actually judged, not the whole body.
        best_chunk_pos: Its character offset, used to anchor snippets.
        score: Final ranking score: blended RRF position and rerank
            relevance, or plain `1 / rank` when reranking was skipped.
        context: Hierarchical context for this path, or None.
        docid: Six-char hash prefix.
        explain: Score trace when `QueryOptions.explain` was set.
    """

    file: str
    display_path: str
    title: str
    body: str
    best_chunk: str
    best_chunk_pos: int
    score: float
    context: str | None
    docid: str
    explain: HybridQueryExplain | None = None


def _pick_best_chunk(
    chunks: list[tuple[str, int]], query_terms: list[str], intent_terms: list[str]
) -> tuple[str, int]:
    best_idx = 0
    best_score = -1.0
    for i, (chunk_text, _pos) in enumerate(chunks):
        chunk_lower = chunk_text.lower()
        score = float(sum(1 for t in query_terms if t in chunk_lower))
        score += sum(0.5 for t in intent_terms if t in chunk_lower)
        if score > best_score:
            best_score = score
            best_idx = i
    return chunks[best_idx]


@dataclass(frozen=True)
class ModelConfig:
    """The three router model slugs the hybrid pipeline needs. They always
    travel together and always come from the same settings object, so
    every call site used to thread the same three arguments through by
    hand."""

    embed: str
    generate: str
    rerank: str

    @classmethod
    def from_settings(cls, settings: Settings) -> ModelConfig:
        """Build from application settings - the usual construction path."""
        return cls(
            embed=settings.embed_model,
            generate=settings.generate_model,
            rerank=settings.rerank_model,
        )


@dataclass(frozen=True)
class QueryOptions:
    """Tunables for one hybrid query. Defaults match the previous
    per-parameter defaults exactly, so `QueryOptions()` is the old
    all-defaults call.

    Bundled so adding a knob doesn't change `hybrid_query`'s signature and
    every caller in turn - the CLI `query` command, the MCP `query` tool,
    and the benchmark runner all build one of these.
    """

    limit: int = 10
    min_score: float = 0.0
    candidate_limit: int = RERANK_CANDIDATE_LIMIT
    collection_name: str | None = None
    intent: str | None = None
    skip_rerank: bool = False
    explain: bool = False
    preexpanded: list[ExpandedQuery] | None = None
    """Typed sub-queries the caller spelled out (the `lex:`/`vec:`/`hyde:`
    document syntax, or the MCP tool's `searches`). When set, the BM25
    strong-signal probe and `expand_query()` are skipped entirely, and -
    since there's no single canonical "original" query left once the
    caller has enumerated the sub-queries - no list gets the 2x
    "original" RRF weight; every list is explicit and weighted 1x."""


async def hybrid_query(
    session: AsyncSession,
    user: CurrentUser,
    query: str,
    llm_client: LlmClient,
    models: ModelConfig,
    options: QueryOptions | None = None,
) -> list[HybridQueryResult]:
    """BM25 + vector + query expansion + RRF + chunked reranking - port of
    the TS reference's `hybridQuery`. See `QueryOptions` for the tunables.
    """
    opts = options or QueryOptions()
    # Unpacked once rather than referenced as `opts.x` throughout: keeps
    # the pipeline below reading as it did before the options object.
    embed_model, generate_model, rerank_model = models.embed, models.generate, models.rerank
    limit, min_score = opts.limit, opts.min_score
    candidate_limit, collection_name = opts.candidate_limit, opts.collection_name
    intent, skip_rerank = opts.intent, opts.skip_rerank
    explain, preexpanded = opts.explain, opts.preexpanded

    ranked_lists: list[list[RankedResult]] = []
    query_types: list[str] = []
    docid_map: dict[str, str] = {}
    context_map: dict[str, str | None] = {}

    def _remember(results: list[Any]) -> list[RankedResult]:
        ranked = []
        for r in results:
            docid_map[r.filepath] = r.docid
            context_map[r.filepath] = r.context
            ranked.append(RankedResult(r.filepath, r.display_path, r.title, r.body, r.score))
        return ranked

    initial_fts: list[Any] = []
    if preexpanded is not None:
        expanded = preexpanded
    else:
        initial_fts = await search_fts(
            session, user, query, limit=20, collection_name=collection_name
        )
        top_score = initial_fts[0].score if initial_fts else 0.0
        second_score = initial_fts[1].score if len(initial_fts) > 1 else 0.0
        has_strong_signal = (
            not intent
            and bool(initial_fts)
            and top_score >= STRONG_SIGNAL_MIN_SCORE
            and (top_score - second_score) >= STRONG_SIGNAL_MIN_GAP
        )
        expanded = (
            []
            if has_strong_signal
            else await expand_query(llm_client, query, generate_model, intent)
        )

    if initial_fts:
        ranked_lists.append(_remember(initial_fts))
        query_types.append("original")

    for q in expanded:
        if q.type != "lex":
            continue
        fts_results = await search_fts(
            session, user, q.query, limit=20, collection_name=collection_name
        )
        if fts_results:
            ranked_lists.append(_remember(fts_results))
            query_types.append("lex")

    vec_queries: list[tuple[str, str]] = [] if preexpanded is not None else [("original", query)]
    vec_queries.extend((q.type, q.query) for q in expanded if q.type in ("vec", "hyde"))

    for query_type, text in vec_queries:
        vec_results = await search_vec(
            session, user, text, llm_client, embed_model, limit=20, collection_name=collection_name
        )
        if vec_results:
            ranked_lists.append(_remember(vec_results))
            query_types.append(query_type)

    weights = _hybrid_rrf_weights(query_types)
    fused = reciprocal_rank_fusion(ranked_lists, weights)
    candidates = fused[:candidate_limit]
    if not candidates:
        return []

    query_terms = [t for t in query.lower().split() if len(t) > 2]
    intent_terms = extract_intent_terms(intent) if intent else []

    chunk_info: dict[str, tuple[str, int]] = {}
    for cand in candidates:
        chunks = chunk_document(cand.body)
        if chunks:
            chunk_info[cand.file] = _pick_best_chunk(chunks, query_terms, intent_terms)

    rrf_rank_map = {c.file: i + 1 for i, c in enumerate(candidates)}
    candidate_map = {c.file: c for c in candidates}

    def _rrf_only_results() -> list[HybridQueryResult]:
        """Rank purely by fused RRF position (1/rank), no rerank pass.

        Backs both `skip_rerank` and the fallback when the rerank round
        trip fails - the candidates are already retrieved and fused at
        that point, so degraded ordering beats no results at all.
        """
        results = []
        for filepath, rrf_rank in rrf_rank_map.items():
            score = 1 / rrf_rank
            if score < min_score:
                continue
            cand = candidate_map[filepath]
            best_chunk, best_pos = chunk_info.get(filepath, (cand.body, 0))
            results.append(
                HybridQueryResult(
                    file=filepath,
                    display_path=cand.display_path,
                    title=cand.title,
                    body=cand.body,
                    best_chunk=best_chunk,
                    best_chunk_pos=best_pos,
                    score=score,
                    context=context_map.get(filepath),
                    docid=docid_map.get(filepath, ""),
                    explain=(
                        HybridQueryExplain(
                            rrf=RrfExplain(rrf_rank, 1.0, score),
                            rerank_score=0.0,
                            blended_score=score,
                        )
                        if explain
                        else None
                    ),
                )
            )
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:limit]

    if skip_rerank:
        return _rrf_only_results()

    chunks_to_rerank = [
        (cand.file, chunk_info[cand.file][0]) for cand in candidates if cand.file in chunk_info
    ]
    if not chunks_to_rerank:
        return []

    rerank_query = f"{intent}\n\n{query}" if intent else query
    try:
        # Both calls hit the router: /tokenize (for the per-pair token
        # budget) and /rerank. Either failing used to abort the whole
        # query, even though the fused candidates were already in hand -
        # the same misbehavior expand_query() degrades on, so degrade the
        # same way here rather than losing good-enough results.
        doc_texts = [
            await _rerank_safe_text(llm_client, text, rerank_model) for _, text in chunks_to_rerank
        ]
        rerank_scores = await llm_client.rerank(rerank_query, doc_texts, rerank_model)
    except (httpx.HTTPError, IndexError, KeyError, ValueError, TypeError):
        return _rrf_only_results()

    blended = []
    for (filepath, _text), rerank_score in zip(chunks_to_rerank, rerank_scores, strict=True):
        rrf_rank = rrf_rank_map.get(filepath, candidate_limit)
        rrf_weight = 0.75 if rrf_rank <= 3 else (0.60 if rrf_rank <= 10 else 0.40)
        rrf_position_score = 1 / rrf_rank
        blended_score = rrf_weight * rrf_position_score + (1 - rrf_weight) * rerank_score

        cand = candidate_map[filepath]
        best_chunk, best_pos = chunk_info.get(filepath, (cand.body, 0))
        blended.append(
            HybridQueryResult(
                file=filepath,
                display_path=cand.display_path,
                title=cand.title,
                body=cand.body,
                best_chunk=best_chunk,
                best_chunk_pos=best_pos,
                score=blended_score,
                context=context_map.get(filepath),
                docid=docid_map.get(filepath, ""),
                explain=(
                    HybridQueryExplain(
                        rrf=RrfExplain(rrf_rank, rrf_weight, rrf_position_score),
                        rerank_score=rerank_score,
                        blended_score=blended_score,
                    )
                    if explain
                    else None
                ),
            )
        )

    blended.sort(key=lambda r: r.score, reverse=True)
    return [r for r in blended if r.score >= min_score][:limit]
