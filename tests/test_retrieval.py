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
    result = await find_document(session, user, "marq://docs/notes/alpha.md")
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
    virtual path (marq://collection/path), not just the bare document
    path, so a "collection/path" display-form lookup (as printed by
    search/ls/multi-get) resolves too."""
    await _seed(session, user)
    result = await find_document(session, user, "docs/notes/alpha.md")
    assert isinstance(result, DocumentDetail)
    assert result.display_path == "docs/notes/alpha.md"


async def test_find_document_by_bare_filename_suffix(
    session: AsyncSession, user: CurrentUser
) -> None:
    """A bare filename is a valid partial path - the suffix match resolves
    it at the segment boundary."""
    await _seed(session, user)
    result = await find_document(session, user, "alpha.md")
    assert isinstance(result, DocumentDetail)
    assert result.display_path == "docs/notes/alpha.md"


async def test_find_document_suffix_match_stops_at_segment_boundary(
    session: AsyncSession, user: CurrentUser
) -> None:
    """Regression for the third review's finding 4: the suffix match used
    a bare endswith, so `pha.md` silently resolved to notes/alpha.md
    mid-filename when no file was literally named that. A partial path
    must only match whole trailing segments; the mid-filename form is an
    honest miss."""
    await _seed(session, user)
    result = await find_document(session, user, "pha.md")
    assert isinstance(result, DocumentNotFound)


async def test_find_document_by_docid(session: AsyncSession, user: CurrentUser) -> None:
    await _seed(session, user)
    result = await find_document(session, user, "#a1a1a1a1a1")
    assert isinstance(result, DocumentDetail)
    assert result.hash == "a1a1a1a1a1"


async def test_find_document_by_colliding_docid_is_deterministic(
    session: AsyncSession, user: CurrentUser
) -> None:
    """A 6-char docid can front two documents. Which one wins is arbitrary,
    but it must be the *same* one every call - `_active_document_refs` is
    ordered so the first-match-wins scan can't depend on Postgres's row
    order."""
    collection_id = await _seed(session, user)
    for suffix, path in (("1", "notes/collide-b.md"), ("2", "notes/collide-a.md")):
        digest = f"ffffff{suffix}"
        await insert_content(session, digest, f"colliding body {suffix}")
        await insert_document(
            session, collection_id, path, f"Collide {suffix}", digest, utcnow(), utcnow()
        )
    await session.commit()

    resolved = []
    for _ in range(4):
        result = await find_document(session, user, "#ffffff")
        assert isinstance(result, DocumentDetail)
        resolved.append(result.display_path)

    assert len(set(resolved)) == 1
    # Ordered by (collection name, path), so the alphabetically first path wins.
    assert resolved[0] == "docs/notes/collide-a.md"


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


async def test_multi_get_mixed_comma_and_glob_matches_nothing(
    session: AsyncSession, user: CurrentUser
) -> None:
    """Pins the documented parity quirk: the two forms don't combine. A
    pattern holding any glob metacharacter is treated wholly as a glob, so
    the comma stays a literal and matches no path."""
    await _seed(session, user)
    results = await multi_get(session, user, "notes/alpha.md,notes/b*.md", line_numbers=False)
    assert results == []


async def test_multi_get_bracket_counts_as_glob_in_comma_heuristic(
    session: AsyncSession, user: CurrentUser
) -> None:
    """Regression: the comma-vs-glob heuristic checked only `*?{`, so a
    pattern like `a[l]pha.md,beta.md` was misclassified as a comma list
    even though _matches_pattern honours fnmatch's `[seq]`. It must count
    as one glob (whose literal comma matches nothing), like `*` does."""
    await _seed(session, user)
    results = await multi_get(
        session, user, "notes/a[l]pha.md,notes/beta.md", line_numbers=False
    )
    assert results == []
    # And without a comma, [seq] works as a plain glob.
    results = await multi_get(session, user, "notes/[ab]*.md", line_numbers=False)
    assert {r.display_path for r in results} == {"docs/notes/alpha.md", "docs/notes/beta.md"}


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


async def test_list_files_prefix_is_literal_not_a_like_pattern(
    session: AsyncSession, user: CurrentUser
) -> None:
    """Regression: the prefix was interpolated into LIKE unescaped, so
    `_`/`%` acted as wildcards - `ls docs/s_c` listed all of `src/`."""
    collection_id = await _seed(session, user)
    for digest, path in (("e5e5e5e5e5", "s_c/one.md"), ("f6f6f6f6f6", "src/two.md")):
        await insert_content(session, digest, f"body of {path}")
        await insert_document(session, collection_id, path, path, digest, utcnow(), utcnow())
    await session.commit()

    files = await list_files(session, user, "docs", "s_c")
    assert {f.path for f in files} == {"s_c/one.md"}


async def test_get_status_reports_doc_counts(session: AsyncSession, user: CurrentUser) -> None:
    await _seed(session, user)
    status = await get_status(session, user)
    assert status.total_documents == 2
    assert [c.name for c in status.collections] == ["docs"]
    assert status.collections[0].doc_count == 2
