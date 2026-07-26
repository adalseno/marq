"""Phase 6: document retrieval (get/multi-get/ls/status) tests - find_document,
multi_get, match_files_by_glob, list_files, get_status - against a real
scratch Postgres schema (see conftest.py).
"""

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from qmd_py.auth import CurrentUser
from qmd_py.store import (
    DEFAULT_MULTI_GET_MAX_BYTES,
    DocumentDetail,
    DocumentNotFound,
    add_collection,
    find_document,
    get_status,
    insert_content,
    insert_document,
    list_files,
    match_files_by_glob,
    multi_get,
    utcnow,
)

pytestmark = pytest.mark.integration


async def _seed(session: AsyncSession, user: CurrentUser) -> int:
    collection = await add_collection(session, user, "docs", "/tmp/docs")
    await insert_content(session, "a1a1a1a1a1", "alpha body content")
    await insert_document(
        session, collection.id, "notes/alpha.md", "Alpha", "a1a1a1a1a1", utcnow(), utcnow()
    )
    await insert_content(session, "b2b2b2b2b2", "beta body content")
    await insert_document(
        session, collection.id, "notes/beta.md", "Beta", "b2b2b2b2b2", utcnow(), utcnow()
    )
    await session.commit()
    return collection.id


async def test_find_document_by_exact_virtual_path(
    session: AsyncSession, user: CurrentUser
) -> None:
    await _seed(session, user)
    result = await find_document(session, user, "qmd://docs/notes/alpha.md")
    assert isinstance(result, DocumentDetail)
    assert result.display_path == "docs/notes/alpha.md"
    assert result.body == "alpha body content"


async def test_find_document_by_bare_relative_path(
    session: AsyncSession, user: CurrentUser
) -> None:
    await _seed(session, user)
    result = await find_document(session, user, "notes/alpha.md")
    assert isinstance(result, DocumentDetail)
    assert result.display_path == "docs/notes/alpha.md"


async def test_find_document_by_collection_prefixed_display_path(
    session: AsyncSession, user: CurrentUser
) -> None:
    """Regression test: the suffix match must compare against the full
    virtual path (qmd://collection/path), not just the bare document
    path, so a "collection/path" display-form lookup (as printed by
    search/ls/multi-get) resolves too."""
    await _seed(session, user)
    result = await find_document(session, user, "docs/notes/alpha.md")
    assert isinstance(result, DocumentDetail)
    assert result.display_path == "docs/notes/alpha.md"


async def test_find_document_by_docid(session: AsyncSession, user: CurrentUser) -> None:
    await _seed(session, user)
    result = await find_document(session, user, "#a1a1a1a1a1")
    assert isinstance(result, DocumentDetail)
    assert result.hash == "a1a1a1a1a1"


async def test_find_document_not_found_reports_similar_files(
    session: AsyncSession, user: CurrentUser
) -> None:
    await _seed(session, user)
    result = await find_document(session, user, "notes/alphaa.md")
    assert isinstance(result, DocumentNotFound)
    assert "notes/alpha.md" in result.similar_files


async def test_match_files_by_glob(session: AsyncSession, user: CurrentUser) -> None:
    await _seed(session, user)
    matches = await match_files_by_glob(session, user, "notes/*.md")
    assert {m.display_path for m in matches} == {"notes/alpha.md", "notes/beta.md"}


async def test_multi_get_comma_list(session: AsyncSession, user: CurrentUser) -> None:
    await _seed(session, user)
    results = await multi_get(session, user, "notes/alpha.md, notes/beta.md", line_numbers=False)
    assert {r.display_path for r in results} == {"docs/notes/alpha.md", "docs/notes/beta.md"}
    assert all(not r.skipped for r in results)


async def test_multi_get_glob(session: AsyncSession, user: CurrentUser) -> None:
    await _seed(session, user)
    results = await multi_get(session, user, "notes/*.md", line_numbers=False)
    assert len(results) == 2


async def test_multi_get_skips_oversized_files(session: AsyncSession, user: CurrentUser) -> None:
    collection_id = await _seed(session, user)
    big_body = "x" * (DEFAULT_MULTI_GET_MAX_BYTES + 1)
    await insert_content(session, "c3c3c3c3c3", big_body)
    await insert_document(
        session, collection_id, "notes/big.md", "Big", "c3c3c3c3c3", utcnow(), utcnow()
    )
    await session.commit()

    results = await multi_get(session, user, "notes/big.md")
    assert len(results) == 1
    assert results[0].skipped is True
    assert "too large" in (results[0].skip_reason or "")


async def test_multi_get_truncates_and_adds_line_numbers(
    session: AsyncSession, user: CurrentUser
) -> None:
    collection_id = await _seed(session, user)
    body = "\n".join(f"line{i}" for i in range(10))
    await insert_content(session, "d4d4d4d4d4", body)
    await insert_document(
        session, collection_id, "notes/lines.md", "Lines", "d4d4d4d4d4", utcnow(), utcnow()
    )
    await session.commit()

    results = await multi_get(session, user, "notes/lines.md", max_lines=3)
    assert len(results) == 1
    assert "truncated 7 more lines" in results[0].body
    assert results[0].body.startswith("1: line0")


async def test_list_files_under_path_prefix(session: AsyncSession, user: CurrentUser) -> None:
    await _seed(session, user)
    files = await list_files(session, user, "docs", "notes")
    assert {f.path for f in files} == {"notes/alpha.md", "notes/beta.md"}


async def test_get_status_reports_doc_counts(session: AsyncSession, user: CurrentUser) -> None:
    await _seed(session, user)
    status = await get_status(session, user)
    assert status.total_documents == 2
    assert [c.name for c in status.collections] == ["docs"]
    assert status.collections[0].doc_count == 2
