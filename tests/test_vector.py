"""Phase 5: chunking/table-naming unit tests + embed_pending_documents/
search_vec integration tests against a real scratch Postgres schema and
the real llama.cpp router (MARQ_LLM_BASE_URL) - there's no local/mock LLM
story for this project (see the plan's "pure HTTP client" decision), so
these necessarily hit the real embedding endpoint.
"""

from collections.abc import AsyncIterator

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from qmd_py.auth import CurrentUser
from qmd_py.config import get_settings
from qmd_py.llm.client import LlmClient
from qmd_py.search.vector import (
    CHUNK_OVERLAP_TOKENS,
    CHUNK_SIZE_TOKENS,
    chunk_document,
    embed_pending_documents,
    embeddings_table_name,
    search_vec,
)
from qmd_py.store import add_collection, insert_content, insert_document, utcnow

EMBED_MODEL = "bge-m3-q8_0"
EMBED_DIM = 1024


def test_embeddings_table_name_sanitizes_slug() -> None:
    assert embeddings_table_name("bge-m3-q8_0") == "embeddings_bge_m3_q8_0"


def test_short_document_is_a_single_chunk() -> None:
    chunks = chunk_document("hello world")
    assert chunks == [("hello world", 0)]


def test_long_document_is_split_with_overlap() -> None:
    body = "x" * 5000
    chunks = chunk_document(body)
    assert len(chunks) > 1
    # consecutive chunks overlap: second chunk's start position is before
    # the first chunk's end
    first_text, first_pos = chunks[0]
    _, second_pos = chunks[1]
    assert second_pos < first_pos + len(first_text)


def test_empty_document_is_one_empty_chunk() -> None:
    """Not zero chunks: embed_pending_documents zips chunks against the
    embedding vectors with strict=True, so an empty list here would mean a
    document silently never gets embedded."""
    assert chunk_document("") == [("", 0)]


def test_document_exactly_at_the_chunk_boundary_stays_one_chunk() -> None:
    max_chars = CHUNK_SIZE_TOKENS * 3
    assert chunk_document("x" * max_chars) == [("x" * max_chars, 0)]


def test_document_one_char_over_the_boundary_splits() -> None:
    max_chars = CHUNK_SIZE_TOKENS * 3
    chunks = chunk_document("x" * (max_chars + 1))

    assert len(chunks) == 2
    assert chunks[0][1] == 0


def test_chunk_positions_advance_by_size_minus_overlap() -> None:
    max_chars = CHUNK_SIZE_TOKENS * 3
    overlap_chars = CHUNK_OVERLAP_TOKENS * 3
    chunks = chunk_document("x" * (max_chars * 3))

    step = max_chars - overlap_chars
    assert [pos for _, pos in chunks[:3]] == [0, step, step * 2]


def test_every_chunk_position_indexes_its_own_text() -> None:
    body = "".join(f"{i:04d}" for i in range(2000))
    for text, pos in chunk_document(body):
        assert body[pos : pos + len(text)] == text


def test_last_chunk_reaches_the_end_of_the_body() -> None:
    body = "y" * (CHUNK_SIZE_TOKENS * 3 * 2 + 17)
    text, pos = chunk_document(body)[-1]

    assert pos + len(text) == len(body)


def test_embeddings_table_name_collapses_runs_of_punctuation() -> None:
    assert embeddings_table_name("Qwen3::Embedding--0.6B") == "embeddings_qwen3_embedding_0_6b"


@pytest.fixture
async def llm_client() -> AsyncIterator[LlmClient]:
    client = LlmClient(get_settings().llm_base_url)
    yield client
    await client.aclose()


@pytest.mark.integration
async def test_embed_and_search_vec_finds_semantically_related_doc(
    session: AsyncSession, user: CurrentUser, llm_client: LlmClient
) -> None:
    collection = await add_collection(session, user, "vecdocs", "/tmp/vecdocs")
    await insert_content(
        session, "vhash1", "The cat sat on the mat and purred contentedly in the sun."
    )
    await insert_document(
        session, collection.id, "cat.md", "A Cat Story", "vhash1", utcnow(), utcnow()
    )
    await insert_content(
        session, "vhash2", "Quarterly revenue projections for the fiscal year budget."
    )
    await insert_document(
        session, collection.id, "budget.md", "Budget Report", "vhash2", utcnow(), utcnow()
    )
    await session.commit()

    embed_result = await embed_pending_documents(
        session, user, llm_client, EMBED_MODEL, EMBED_DIM
    )
    await session.commit()
    assert embed_result.docs_processed == 2
    assert embed_result.chunks_embedded == 2

    results = await search_vec(
        session, user, "a feline resting in sunlight", llm_client, EMBED_MODEL
    )
    assert len(results) == 2
    assert results[0].display_path == "vecdocs/cat.md"
    assert results[0].source == "vec"
    assert results[0].score > results[1].score


@pytest.mark.integration
async def test_search_vec_respects_collection_filter(
    session: AsyncSession, user: CurrentUser, llm_client: LlmClient
) -> None:
    coll_a = await add_collection(session, user, "veca", "/tmp/veca")
    coll_b = await add_collection(session, user, "vecb", "/tmp/vecb")
    await insert_content(session, "vhasha", "Deep learning neural networks and gradient descent.")
    await insert_document(session, coll_a.id, "a.md", "A", "vhasha", utcnow(), utcnow())
    await insert_content(session, "vhashb", "Deep learning neural networks and gradient descent.")
    await insert_document(session, coll_b.id, "b.md", "B", "vhashb", utcnow(), utcnow())
    await session.commit()

    await embed_pending_documents(session, user, llm_client, EMBED_MODEL, EMBED_DIM)
    await session.commit()

    scoped = await search_vec(
        session, user, "machine learning", llm_client, EMBED_MODEL, collection_name="veca"
    )
    assert [r.collection_name for r in scoped] == ["veca"]


@pytest.mark.integration
async def test_embed_pending_documents_is_idempotent(
    session: AsyncSession, user: CurrentUser, llm_client: LlmClient
) -> None:
    collection = await add_collection(session, user, "vecidem", "/tmp/vecidem")
    await insert_content(session, "vhashc", "Some content about gardening and vegetables.")
    await insert_document(session, collection.id, "c.md", "C", "vhashc", utcnow(), utcnow())
    await session.commit()

    first = await embed_pending_documents(session, user, llm_client, EMBED_MODEL, EMBED_DIM)
    await session.commit()
    assert first.docs_processed == 1

    second = await embed_pending_documents(session, user, llm_client, EMBED_MODEL, EMBED_DIM)
    await session.commit()
    assert second.docs_processed == 0


@pytest.mark.integration
async def test_search_vec_returns_empty_when_nothing_embedded_yet(
    session: AsyncSession, user: CurrentUser, llm_client: LlmClient
) -> None:
    """Regression test: before anything has ever been embedded with a
    model, its embeddings_<slug> table doesn't exist yet - search_vec
    must return [] gracefully (matching the TS reference's
    hasVectorColumn() guard), not raise a raw "relation does not exist"
    SQL error."""
    collection = await add_collection(session, user, "vecempty", "/tmp/vecempty")
    await insert_content(session, "vhashd", "Some content never embedded.")
    await insert_document(session, collection.id, "d.md", "D", "vhashd", utcnow(), utcnow())
    await session.commit()

    results = await search_vec(session, user, "anything", llm_client, EMBED_MODEL)
    assert results == []
