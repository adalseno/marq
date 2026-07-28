"""Vector search: pgvector embeddings, one physical table per embedding
model (`embeddings_<slug>`), created dynamically once a model's dimension
is known - fixes the TS reference's one-model-at-a-time limitation (a
single `content_vectors.embedding` column, destructively dropped on a
model switch). Candidate retrieval happens in a CTE via the HNSW-indexed
`<=>` operator, exactly as the TS reference's `searchVec` does; the
collection filter is applied only on the outer join, never inside the
CTE, since `ORDER BY distance LIMIT n` is the only access pattern that
uses the ANN index.
"""

import re
from dataclasses import dataclass
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col, select

from qmd_py.auth import CurrentUser
from qmd_py.db.models import EmbeddingModel
from qmd_py.db.result import affected_rows
from qmd_py.llm.client import LlmClient, format_doc_for_embedding, format_query_for_embedding
from qmd_py.search._acl import collection_names_by_id, resolve_collection_ids
from qmd_py.search.fts import SearchResult, get_context_for_path, get_docid

# Chunking: 900 tokens/chunk, 15% overlap - fixed character-window
# slicing with a conservative chars-per-token estimate. This is a
# SIMPLIFICATION of the TS reference's chunkDocumentByTokens (no smart
# heading/AST break-point search, no tokenizer-verified re-splitting):
# good enough for vsearch's mechanics (Phase 5's scope - candidate CTE,
# dedup, HNSW). Revisit for exact chunk-boundary parity when Phase 8's
# bench/quality work needs it.
CHUNK_SIZE_TOKENS = 900
CHUNK_OVERLAP_TOKENS = int(CHUNK_SIZE_TOKENS * 0.15)
_AVG_CHARS_PER_TOKEN = 3


def chunk_document(body: str) -> list[tuple[str, int]]:
    """Split a body into overlapping character windows for embedding.

    Fixed-width slicing on a conservative chars-per-token estimate - no
    heading or AST-aware break points, and no tokenizer verification.
    That is a deliberate simplification of the TS reference; the estimate
    can undercount dense code, which is why the rerank path re-measures
    with the real tokenizer (see `hybrid._rerank_safe_text`).

    Returns:
        `(text, start_offset)` pairs, always at least one - an empty body
        yields `[("", 0)]` rather than an empty list, so a document can
        never silently go unembedded. Offsets index into `body`, and
        consecutive chunks overlap by design so a match spanning a
        boundary is still retrievable.
    """
    max_chars = CHUNK_SIZE_TOKENS * _AVG_CHARS_PER_TOKEN
    overlap_chars = CHUNK_OVERLAP_TOKENS * _AVG_CHARS_PER_TOKEN

    if len(body) <= max_chars:
        return [(body, 0)]

    chunks: list[tuple[str, int]] = []
    step = max_chars - overlap_chars
    pos = 0
    while pos < len(body):
        chunks.append((body[pos : pos + max_chars], pos))
        if pos + max_chars >= len(body):
            break
        pos += step
    return chunks


_SLUG_SANITIZE = re.compile(r"[^a-z0-9_]+")


def embeddings_table_name(model_slug: str) -> str:
    """Physical table name holding one embedding model's vectors.

    Each model gets its own table, so adding a model is additive and
    switching between them never drops data.

    The slug is reduced to lowercase alphanumerics and underscores, which
    is what makes it safe to interpolate into DDL - these names cannot be
    bound as parameters.
    """
    sanitized = _SLUG_SANITIZE.sub("_", model_slug.lower()).strip("_")
    return f"embeddings_{sanitized}"


async def _embeddings_table_exists(session: AsyncSession, table_name: str) -> bool:
    result = await session.execute(text("SELECT to_regclass(:name)"), {"name": table_name})
    return result.scalar_one() is not None


async def has_embeddings_table(session: AsyncSession, model_slug: str) -> bool:
    """Whether anything has ever been embedded with this model - i.e. its
    `embeddings_<slug>` table exists. Lets a caller batching query
    embeddings up front (see `hybrid.hybrid_query`) skip the embedding
    round trip entirely when vector search would return empty anyway."""
    return await _embeddings_table_exists(session, embeddings_table_name(model_slug))


async def _pgvector_schema(session: AsyncSession) -> str:
    """Schema pgvector's `vector` type/opclasses actually live in - looked
    up from `pg_extension` rather than assumed to be the app's own
    configured schema. On a fresh database, Alembic's `CREATE EXTENSION IF
    NOT EXISTS vector` does install it there - but on a shared database
    (e.g. this project's real server, which already had another service
    install pgvector into `public` long before qmd-py existed), extensions
    are database-wide singletons: Alembic's call just found it already
    installed elsewhere and no-opped. See `ensure_embedding_model` for why
    every reference to the type/opclasses is schema-qualified rather than
    relying on search_path.
    """
    result = await session.execute(
        text(
            "SELECT n.nspname FROM pg_extension e "
            "JOIN pg_namespace n ON n.oid = e.extnamespace "
            "WHERE e.extname = 'vector'"
        )
    )
    schema = result.scalar_one_or_none()
    if schema is None:
        raise RuntimeError("pgvector extension ('vector') is not installed in this database")
    return str(schema)


async def get_or_probe_dimension(
    session: AsyncSession, llm_client: LlmClient, model_slug: str
) -> int:
    """The registered dimension for an already-known model, or one
    embedding call to discover it for a brand-new one - avoids wasting a
    probe request every time `embed` runs against a model already in use."""
    existing = (
        await session.execute(
            select(col(EmbeddingModel.dimension)).where(col(EmbeddingModel.slug) == model_slug)
        )
    ).scalar_one_or_none()
    if existing is not None:
        return existing
    probe = await llm_client.embed(["dimension probe"], model_slug)
    return len(probe[0])


async def ensure_embedding_model(
    session: AsyncSession, slug: str, role: str, dimension: int
) -> EmbeddingModel:
    """Registers (or fetches) an embedding-model row and creates its
    dedicated `embeddings_<slug>` table - idempotent, never destructive;
    switching models means a different table, never a dropped column."""
    existing = (
        await session.execute(select(EmbeddingModel).where(col(EmbeddingModel.slug) == slug))
    ).scalar_one_or_none()
    if existing is None:
        existing = EmbeddingModel(slug=slug, role=role, dimension=dimension)
        session.add(existing)
        await session.flush()

    table_name = embeddings_table_name(slug)
    # pgvector's `vector` type/opclasses are database-wide singletons,
    # installed into whichever schema actually created the extension (see
    # `_pgvector_schema`) - schema-qualified here so they resolve correctly
    # even from a connection whose search_path is some *other* schema
    # entirely (e.g. a test harness's isolated scratch schema), without
    # needing that schema on the search_path at all (see tests/conftest.py's
    # `_schema_url` for why that's dangerous: it would make CREATE TABLE IF
    # NOT EXISTS treat this app's own tables as already existing there too,
    # and silently leak test data into it).
    pgvector_schema = await _pgvector_schema(session)
    await session.execute(
        text(
            f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                hash TEXT NOT NULL REFERENCES content(hash) ON DELETE CASCADE,
                seq INTEGER NOT NULL DEFAULT 0,
                pos INTEGER NOT NULL DEFAULT 0,
                total_chunks INTEGER NOT NULL DEFAULT 1,
                embedded_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                embedding {pgvector_schema}.vector({dimension}) NOT NULL,
                PRIMARY KEY (hash, seq)
            )
            """
        )
    )
    await session.execute(
        text(
            f"CREATE INDEX IF NOT EXISTS idx_{table_name}_embedding "
            f"ON {table_name} USING hnsw (embedding {pgvector_schema}.vector_cosine_ops)"
        )
    )
    return existing


def _to_vector_literal(embedding: list[float]) -> str:
    return "[" + ",".join(repr(float(v)) for v in embedding) + "]"


@dataclass
class EmbedResult:
    """Counts from one `embed_pending_documents()` pass.

    Attributes:
        docs_processed: Distinct content hashes embedded. Documents that
            already had vectors for this model are skipped and not
            counted.
        chunks_embedded: Vectors written. Higher than `docs_processed`
            whenever documents were long enough to split.
    """

    docs_processed: int
    chunks_embedded: int


async def embed_pending_documents(
    session: AsyncSession,
    user: CurrentUser,
    llm_client: LlmClient,
    model_slug: str,
    dimension: int,
    collection_name: str | None = None,
) -> EmbedResult:
    """Embed every active document not yet present in `embeddings_<slug>`.

    Commits after each document rather than leaving one giant transaction
    to the caller: the pending query (`hash NOT IN (...)`) makes the run
    naturally resumable, so a crash midway through a large embed keeps
    everything already written, and the transaction stays short on the
    shared server. No retry/duration-cap/force-re-embed yet.
    """
    await ensure_embedding_model(session, model_slug, "embed", dimension)
    # Committed before the pending scan, not left to the caller: the model
    # row and its table (DDL is transactional in Postgres) must survive
    # even when this run embeds nothing, since the per-document commits
    # below are the only other commit points.
    await session.commit()
    table_name = embeddings_table_name(model_slug)

    collection_ids = await resolve_collection_ids(session, user, collection_name)
    if not collection_ids:
        return EmbedResult(0, 0)

    pending = (
        await session.execute(
            text(
                f"""
                SELECT DISTINCT d.hash, c.doc AS body, d.title
                FROM document d
                JOIN content c ON c.hash = d.hash
                WHERE d.active AND d.collection_id = ANY(:collection_ids)
                  AND d.hash NOT IN (SELECT hash FROM {table_name})
                """
            ),
            {"collection_ids": collection_ids},
        )
    ).all()

    pgvector_schema = await _pgvector_schema(session)
    insert_stmt = text(
        f"""
        INSERT INTO {table_name} (hash, seq, pos, total_chunks, embedding)
        VALUES (:hash, :seq, :pos, :total_chunks, (:vec)::{pgvector_schema}.vector)
        ON CONFLICT (hash, seq) DO UPDATE SET
            pos = excluded.pos, total_chunks = excluded.total_chunks,
            embedding = excluded.embedding, embedded_at = now()
        """
    )

    docs_processed = 0
    chunks_embedded = 0
    for row in pending:
        chunks = chunk_document(row.body)
        texts_to_embed = [
            format_doc_for_embedding(chunk_text, row.title, model_slug)
            for chunk_text, _ in chunks
        ]
        vectors = await llm_client.embed(texts_to_embed, model_slug)

        # One executemany per document instead of one INSERT per chunk.
        await session.execute(
            insert_stmt,
            [
                {
                    "hash": row.hash,
                    "seq": seq,
                    "pos": pos,
                    "total_chunks": len(chunks),
                    "vec": _to_vector_literal(vector),
                }
                for seq, ((_, pos), vector) in enumerate(zip(chunks, vectors, strict=True))
            ],
        )
        await session.commit()
        chunks_embedded += len(chunks)
        docs_processed += 1

    return EmbedResult(docs_processed, chunks_embedded)


@dataclass
class VectorIndexHealth:
    """Whether semantic search is usable, and how stale it is.

    Attributes:
        has_vector_index: Whether this model's table exists at all -
            False means nothing has ever been embedded with it, and
            vector search will return empty rather than fail.
        needs_embedding: Active documents with no vector for this model.
            Equals the total document count when `has_vector_index` is
            False.
    """

    has_vector_index: bool
    needs_embedding: int


async def get_vector_index_health(
    session: AsyncSession, user: CurrentUser, model_slug: str
) -> VectorIndexHealth:
    """Whether `model_slug` has ever been embedded at all, and how many of
    the user's active documents still lack an embedding for it - backs
    the MCP server's dynamic instructions and `status` tool (Phase 9),
    which need this same "needs embedding" count `embed_pending_documents`
    already computes internally but didn't expose on its own."""
    collection_ids = await resolve_collection_ids(session, user, None)
    if not collection_ids:
        return VectorIndexHealth(False, 0)

    table_name = embeddings_table_name(model_slug)
    has_index = await _embeddings_table_exists(session, table_name)
    if not has_index:
        count = (
            await session.execute(
                text(
                    "SELECT COUNT(*) FROM document "
                    "WHERE active AND collection_id = ANY(:collection_ids)"
                ),
                {"collection_ids": collection_ids},
            )
        ).scalar_one()
        return VectorIndexHealth(False, count)

    count = (
        await session.execute(
            text(
                f"""
                SELECT COUNT(DISTINCT d.hash) FROM document d
                WHERE d.active AND d.collection_id = ANY(:collection_ids)
                  AND d.hash NOT IN (SELECT hash FROM {table_name})
                """
            ),
            {"collection_ids": collection_ids},
        )
    ).scalar_one()
    return VectorIndexHealth(True, count)


async def search_vec(
    session: AsyncSession,
    user: CurrentUser,
    query: str,
    llm_client: LlmClient,
    model_slug: str,
    limit: int = 20,
    collection_name: str | None = None,
    query_embedding: list[float] | None = None,
) -> list[SearchResult]:
    """Rank documents by embedding similarity to the query.

    Embeds the query, retrieves nearest chunks through the HNSW index,
    then deduplicates to one hit per document, keeping its closest chunk.

    Args:
        model_slug: Which embedding model's table to search. Must match
            what the documents were embedded with - vectors from
            different models are not comparable.
        limit: Maximum documents returned. Three times this many *chunks*
            are retrieved first, since several may belong to one document.
        collection_name: Restrict to one collection; None searches every
            collection the user can read.
        query_embedding: Precomputed vector for `query`, skipping the
            embedding round trip. Must have been produced from
            `format_query_for_embedding(query, model_slug)` with the same
            model - `hybrid_query` uses this to embed all its vec/hyde
            variants in one batched request.

    Returns:
        Hits ordered by descending similarity, `source="vec"` and
        `chunk_pos` set. Empty - not an error - when nothing has been
        embedded with this model yet, when no collection is accessible,
        or when the index holds no neighbours.

    Note:
        Makes one embedding call to the LLM router per invocation, unless
        `query_embedding` is supplied. The collection filter is applied
        outside the nearest-neighbour CTE, because `ORDER BY distance
        LIMIT n` is the only access pattern the ANN index can serve.
    """
    collection_ids = await resolve_collection_ids(session, user, collection_name)
    if not collection_ids:
        return []

    table_name = embeddings_table_name(model_slug)
    if not await _embeddings_table_exists(session, table_name):
        # Nothing embedded with this model yet - same graceful empty
        # result the TS reference's searchVec gives via hasVectorColumn(),
        # rather than a raw "relation does not exist" SQL error.
        return []

    pgvector_schema = await _pgvector_schema(session)
    if query_embedding is None:
        formatted_query = format_query_for_embedding(query, model_slug)
        query_embedding = (await llm_client.embed([formatted_query], model_slug))[0]

    rows = (
        await session.execute(
            text(
                f"""
                WITH candidates AS (
                    SELECT hash, seq, pos,
                        embedding OPERATOR({pgvector_schema}.<=>) (:vec)::{pgvector_schema}.vector
                            AS distance
                    FROM {table_name}
                    ORDER BY distance ASC
                    LIMIT :candidate_limit
                )
                SELECT
                    c.hash, c.pos, c.distance,
                    d.collection_id, d.path, d.title, d.modified_at,
                    doc.doc AS body
                FROM candidates c
                JOIN document d ON d.hash = c.hash AND d.active
                JOIN content doc ON doc.hash = d.hash
                WHERE d.collection_id = ANY(:collection_ids)
                """
            ),
            {
                "vec": _to_vector_literal(query_embedding),
                # limit*3 candidate pool mirrors the TS reference - the
                # only access pattern that uses the HNSW index.
                "candidate_limit": limit * 3,
                "collection_ids": collection_ids,
            },
        )
    ).all()
    if not rows:
        return []

    # Multiple chunks of the same file can appear - dedupe by
    # (collection_id, path), keeping the closest (lowest-distance) chunk.
    best: dict[tuple[int, str], Any] = {}
    for row in rows:
        key = (row.collection_id, row.path)
        if key not in best or row.distance < best[key].distance:
            best[key] = row

    ranked = sorted(best.values(), key=lambda r: r.distance)[:limit]

    collection_names = await collection_names_by_id(
        session, {r.collection_id for r in ranked}
    )

    results = []
    for row in ranked:
        name = collection_names[row.collection_id]
        context = await get_context_for_path(session, user, row.collection_id, row.path)
        results.append(
            SearchResult(
                filepath=f"marq://{name}/{row.path}",
                display_path=f"{name}/{row.path}",
                title=row.title,
                hash=row.hash,
                docid=get_docid(row.hash),
                collection_name=name,
                modified_at=row.modified_at,
                body_length=len(row.body),
                body=row.body,
                context=context,
                score=1 - row.distance,
                source="vec",
                chunk_pos=row.pos,
            )
        )
    return results


async def cleanup_orphaned_embeddings(session: AsyncSession) -> int:
    """Deletes embedding rows whose hash no longer belongs to any active
    document, across every registered embedding model's table - backs the
    `cleanup` command's "remove orphaned embedding chunks" step. Content
    rows themselves are `cleanup_orphaned_content`'s job (store.py)."""
    model_slugs = (
        await session.execute(select(col(EmbeddingModel.slug)))
    ).scalars().all()

    total = 0
    for slug in model_slugs:
        table_name = embeddings_table_name(slug)
        if not await _embeddings_table_exists(session, table_name):
            continue
        result = await session.execute(
            text(
                f"""
                DELETE FROM {table_name}
                WHERE hash NOT IN (SELECT hash FROM document WHERE active)
                """
            )
        )
        total += affected_rows(result)
    await session.flush()
    return total
