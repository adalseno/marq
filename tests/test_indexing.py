"""Phase 7: reindex_collection tests - the filesystem-walking indexer
backing `collection add` and `update`, against a real scratch Postgres
schema (see conftest.py) and real temp directories.
"""

import logging
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


async def test_reindex_collection_counts_a_title_only_change_as_updated(
    tmp_path: Path, session: AsyncSession, user: CurrentUser
) -> None:
    """Same body hash, different heading: the content row is untouched but
    the document's title (and its search_vector) must be refreshed, so
    this counts as updated rather than unchanged."""
    file_a = tmp_path / "a.md"
    file_a.write_text("# Original Title\n\nstable body")
    collection = await add_collection(session, user, "titles", str(tmp_path), "**/*.md")
    await session.commit()
    await reindex_collection(session, user, "titles")
    await session.commit()

    file_a.write_text("# Renamed Title\n\nstable body")
    result = await reindex_collection(session, user, "titles")
    await session.commit()

    assert result.updated == 1
    assert result.unchanged == 0
    document = await find_active_document(session, collection.id, "a.md")
    assert document is not None
    assert document.title == "Renamed Title"


async def test_reindex_collection_skips_unreadable_and_empty_files(
    tmp_path: Path, session: AsyncSession, user: CurrentUser
) -> None:
    """Non-UTF-8 bytes and whitespace-only files are skipped rather than
    aborting the whole walk."""
    (tmp_path / "good.md").write_text("# Good\n\nreal body")
    (tmp_path / "binary.md").write_bytes(b"\xff\xfe\x00binary garbage")
    (tmp_path / "blank.md").write_text("   \n\n  ")
    collection = await add_collection(session, user, "mixed", str(tmp_path), "**/*.md")
    await session.commit()

    result = await reindex_collection(session, user, "mixed")
    await session.commit()

    assert result.indexed == 1
    assert await get_active_document_paths(session, collection.id) == ["good.md"]


async def test_reindex_collection_survives_a_file_vanishing_between_read_and_stat(
    tmp_path: Path,
    session: AsyncSession,
    user: CurrentUser,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression for the third review's finding 2 (TOCTOU): the `stat()`
    call sat outside the `try` guarding `read_text()`, so a file deleted
    between the two - the fast-moving-worktree race the indexer is
    documented to support - raised an unhandled FileNotFoundError. The
    vanished file must count as skipped instead."""
    import qmd_py.store.indexing as indexing_module

    (tmp_path / "stable.md").write_text("# Stable\n\nbody")
    (tmp_path / "vanishing.md").write_text("# Vanishing\n\nbody")
    collection = await add_collection(session, user, "racy", str(tmp_path), "**/*.md")
    await session.commit()

    # Pin the walk result first: `_discover_files` itself stats every
    # candidate (`is_file()`), which would drop the file before the loop
    # and dodge the race this test exists to simulate. The race is
    # read_text() succeeding and *then* stat() failing.
    monkeypatch.setattr(
        indexing_module, "_discover_files", lambda *a, **k: ["stable.md", "vanishing.md"]
    )
    real_stat = Path.stat

    def racing_stat(self: Path, **kwargs: object) -> object:
        if self.name == "vanishing.md":
            raise FileNotFoundError(f"deleted mid-walk: {self}")
        return real_stat(self, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "stat", racing_stat)
    result = await reindex_collection(session, user, "racy")
    monkeypatch.undo()
    await session.commit()

    assert result.indexed == 1
    assert await get_active_document_paths(session, collection.id) == ["stable.md"]


async def test_reindex_collection_skips_and_counts_oversized_files(
    tmp_path: Path, session: AsyncSession, user: CurrentUser
) -> None:
    """Regression for the third review's finding 1: a distinct-token-heavy
    file over ~1 MB used to reach `to_tsvector` and abort the whole run
    with a raw Postgres "string is too long for tsvector" error. It must
    be soft-skipped and counted instead, and a previously indexed file
    that grows past the cap is deactivated like any other skipped file."""
    (tmp_path / "good.md").write_text("# Good\n\nreal body")
    grower = tmp_path / "grower.md"
    grower.write_text("# Grower\n\nstill small")
    collection = await add_collection(session, user, "big", str(tmp_path), "**/*.md")
    await session.commit()

    first = await reindex_collection(session, user, "big")
    await session.commit()
    assert first.indexed == 2
    assert first.skipped_oversize == 0

    oversized = "# Huge\n\n" + "tok ".join(str(n) for n in range(300_000))
    assert len(oversized) > 1_000_000
    grower.write_text(oversized)
    second = await reindex_collection(session, user, "big")
    await session.commit()

    assert second.skipped_oversize == 1
    assert second.removed == 1
    assert await get_active_document_paths(session, collection.id) == ["good.md"]


async def test_reindex_collection_logs_a_warning_per_skipped_file(
    tmp_path: Path,
    session: AsyncSession,
    user: CurrentUser,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Skipped files are the reindexer's silent degrade path - a document
    missing from search results with no explanation anywhere. Each skip
    logs its path and reason once."""
    (tmp_path / "good.md").write_text("# Good\n\nreal body")
    (tmp_path / "binary.md").write_bytes(b"\xff\xfe\x00binary garbage")
    (tmp_path / "blank.md").write_text("   \n\n  ")
    (tmp_path / "huge.md").write_text("# Huge\n\n" + "tok ".join(str(n) for n in range(300_000)))
    await add_collection(session, user, "noisy", str(tmp_path), "**/*.md")
    await session.commit()

    with caplog.at_level(logging.WARNING, logger="qmd_py.store.indexing"):
        await reindex_collection(session, user, "noisy")
    await session.commit()

    assert "skipping binary.md: UnicodeDecodeError" in caplog.text
    assert "skipping blank.md: file is empty or whitespace-only" in caplog.text
    assert "skipping huge.md" in caplog.text
    assert "exceeds the 1000000-char index limit" in caplog.text
    assert "good.md" not in caplog.text


async def test_reindex_collection_shares_content_between_duplicate_files(
    tmp_path: Path, session: AsyncSession, user: CurrentUser
) -> None:
    """Content is addressed by hash, so two files with identical bodies
    are two documents pointing at one content row."""
    (tmp_path / "one.md").write_text("# Same\n\nidentical body")
    (tmp_path / "two.md").write_text("# Same\n\nidentical body")
    collection = await add_collection(session, user, "dupes", str(tmp_path), "**/*.md")
    await session.commit()

    result = await reindex_collection(session, user, "dupes")
    await session.commit()

    assert result.indexed == 2
    paths = await get_active_document_paths(session, collection.id)
    assert sorted(paths) == ["one.md", "two.md"]
    docs = [await find_active_document(session, collection.id, p) for p in sorted(paths)]
    assert docs[0] is not None and docs[1] is not None
    assert docs[0].hash == docs[1].hash


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
