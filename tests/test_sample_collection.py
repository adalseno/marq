"""End-to-end FTS + vector search tests against the frozen, checked-in
`tests/fixtures/sample-collection/` fixture (see conftest.py's
`sample_collection`) - a small, manageable, always-available multi-file
project (.md/.py/.js/.ts) anyone can run these against without external
data or server access, unlike the book-catalog collection used ad hoc
during development.
"""

from collections.abc import AsyncIterator

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from qmd_py.auth import CurrentUser
from qmd_py.config import get_settings
from qmd_py.db.models import Collection
from qmd_py.llm.client import LlmClient
from qmd_py.search.fts import search_fts
from qmd_py.search.vector import embed_pending_documents, search_vec

pytestmark = pytest.mark.integration

EMBED_MODEL = "bge-m3-q8_0"
EMBED_DIM = 1024


@pytest.fixture
async def llm_client() -> AsyncIterator[LlmClient]:
    client = LlmClient(get_settings().llm_base_url)
    yield client
    await client.aclose()


async def test_fts_finds_the_python_storage_module(
    session: AsyncSession, user: CurrentUser, sample_collection: Collection
) -> None:
    results = await search_fts(session, user, "sqlite migration schema")
    assert results
    assert results[0].display_path == "sample/src/tasks.py"


async def test_fts_finds_the_typescript_types(
    session: AsyncSession, user: CurrentUser, sample_collection: Collection
) -> None:
    results = await search_fts(session, user, "TaskListResponse priority")
    assert results
    assert results[0].display_path == "sample/src/models.ts"


async def test_vector_search_finds_semantically_related_files(
    session: AsyncSession,
    user: CurrentUser,
    sample_collection: Collection,
    llm_client: LlmClient,
) -> None:
    embed_result = await embed_pending_documents(session, user, llm_client, EMBED_MODEL, EMBED_DIM)
    await session.commit()
    assert embed_result.docs_processed == 6

    results = await search_vec(
        session, user, "how are tasks persisted to disk", llm_client, EMBED_MODEL
    )
    assert results
    assert results[0].display_path in {"sample/src/tasks.py", "sample/docs/architecture.md"}


async def test_vector_search_across_the_http_layer(
    session: AsyncSession,
    user: CurrentUser,
    sample_collection: Collection,
    llm_client: LlmClient,
) -> None:
    await embed_pending_documents(session, user, llm_client, EMBED_MODEL, EMBED_DIM)
    await session.commit()

    results = await search_vec(
        session, user, "handling an HTTP request for the task list", llm_client, EMBED_MODEL
    )
    assert results
    assert results[0].display_path == "sample/src/api.js"
