"""Phase 7: reindex_collection tests - the filesystem-walking indexer
backing `collection add` and `update`, against a real scratch Postgres
schema (see conftest.py) and real temp directories.
"""

from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from qmd_py.auth import CurrentUser
from qmd_py.store import (
    add_collection,
    find_active_document,
    get_active_document_paths,
    reindex_collection,
)

pytestmark = pytest.mark.integration


async def test_reindex_collection_indexes_new_files(
    tmp_path: Path, session: AsyncSession, user: CurrentUser
) -> None:
    (tmp_path / "a.md").write_text("# A\n\nbody a")
    (tmp_path / "b.md").write_text("# B\n\nbody b")
    await add_collection(session, user, "proj", str(tmp_path), "**/*.md")
    await session.commit()

    result = await reindex_collection(session, user, "proj")
    await session.commit()
    assert result.indexed == 2
    assert result.updated == 0
    assert result.removed == 0

    second = await reindex_collection(session, user, "proj")
    await session.commit()
    assert second.unchanged == 2
    assert second.indexed == 0


async def test_reindex_collection_detects_changed_and_removed_files(
    tmp_path: Path, session: AsyncSession, user: CurrentUser
) -> None:
    file_a = tmp_path / "a.md"
    file_a.write_text("# A\n\noriginal body")
    await add_collection(session, user, "proj2", str(tmp_path), "**/*.md")
    await session.commit()
    await reindex_collection(session, user, "proj2")
    await session.commit()

    file_a.write_text("# A\n\nchanged body")
    result = await reindex_collection(session, user, "proj2")
    await session.commit()
    assert result.updated == 1

    file_a.unlink()
    result2 = await reindex_collection(session, user, "proj2")
    await session.commit()
    assert result2.removed == 1


async def test_reindex_collection_reactivates_reappeared_file(
    tmp_path: Path, session: AsyncSession, user: CurrentUser
) -> None:
    """Regression test: simulates checking out git branches back and
    forth on the same indexed working tree - a file that disappears then
    reappears must be reactivated, not crash on Document's table-wide
    UNIQUE(collection_id, path) constraint (there's no "inactive" carve-out
    in it, so a second INSERT for the same path is impossible once one
    row - active or not - already exists)."""
    file_a = tmp_path / "a.md"
    file_a.write_text("# A\n\nbody a")
    collection = await add_collection(session, user, "proj3", str(tmp_path), "**/*.md")
    await session.commit()
    await reindex_collection(session, user, "proj3")
    await session.commit()

    file_a.unlink()
    await reindex_collection(session, user, "proj3")
    await session.commit()
    assert await find_active_document(session, collection.id, "a.md") is None

    file_a.write_text("# A\n\nbody a again")
    result = await reindex_collection(session, user, "proj3")
    await session.commit()
    assert result.updated == 1
    assert result.indexed == 0

    reactivated = await find_active_document(session, collection.id, "a.md")
    assert reactivated is not None
    assert await get_active_document_paths(session, collection.id) == ["a.md"]


async def test_reindex_collection_expands_brace_patterns_and_ignores_dirs(
    tmp_path: Path, session: AsyncSession, user: CurrentUser
) -> None:
    (tmp_path / "a.py").write_text("print('a')")
    (tmp_path / "b.md").write_text("# B")
    (tmp_path / "c.txt").write_text("not matched")
    node_modules = tmp_path / "node_modules"
    node_modules.mkdir()
    (node_modules / "ignored.py").write_text("ignored")

    collection = await add_collection(session, user, "proj4", str(tmp_path), "**/*.{py,md}")
    await session.commit()
    await reindex_collection(session, user, "proj4")
    await session.commit()

    paths = set(await get_active_document_paths(session, collection.id))
    assert paths == {"a.py", "b.md"}
