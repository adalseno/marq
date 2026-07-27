"""Phase 3 exit tests: collection/context/content-ingest CRUD through the
service layer, against a real scratch Postgres schema (see conftest.py).
"""

import hashlib
from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from qmd_py.auth import CurrentUser
from qmd_py.db.models import Document, User
from qmd_py.store import (
    _MAX_BRACE_EXPANSIONS,
    CollectionNotFoundError,
    PermissionDeniedError,
    _discover_files,
    _expand_braces,
    _match_named_document,
    add_collection,
    add_context,
    cleanup_orphaned_content,
    context_check,
    deactivate_document,
    extract_title,
    find_active_document,
    get_active_document_paths,
    hash_content,
    insert_content,
    insert_document,
    list_collections,
    list_contexts,
    remove_collection,
    remove_context,
    rename_collection,
    set_global_context,
    update_document,
    utcnow,
)


async def test_hash_content_matches_sha256() -> None:
    assert hash_content("hello") == hashlib.sha256(b"hello").hexdigest()


def test_extract_title_markdown_heading() -> None:
    assert extract_title("# My Title\n\nbody", "doc.md") == "My Title"


def test_extract_title_notes_fallback() -> None:
    content = "# \U0001f4dd Notes\n\n## Real Title\n\nbody"
    assert extract_title(content, "doc.md") == "Real Title"


def test_extract_title_org() -> None:
    assert extract_title("#+TITLE: Org Doc\n\nbody", "doc.org") == "Org Doc"


def test_extract_title_fallback_to_filename() -> None:
    assert extract_title("no heading here", "plain-notes.md") == "plain-notes"


def test_extract_title_org_falls_back_to_first_heading() -> None:
    """No `#+TITLE:` property, so the first `*` outline heading wins."""
    assert extract_title("* First Heading\n\nbody", "doc.org") == "First Heading"


def test_extract_title_org_deep_heading_level() -> None:
    assert extract_title("*** Deep Heading\n", "doc.org") == "Deep Heading"


def test_extract_title_org_without_any_heading_uses_filename() -> None:
    assert extract_title("just body text", "notes.org") == "notes"


def test_extract_title_strips_directories_from_the_filename() -> None:
    assert extract_title("no heading", "deep/nested/file.md") == "file"


def test_extract_title_extensionless_filename() -> None:
    assert extract_title("no heading", "README") == "README"


def test_extract_title_h2_heading_counts() -> None:
    assert extract_title("## Second Level\n\nbody", "doc.md") == "Second Level"


def test_extract_title_notes_quirk_keeps_notes_without_a_successor() -> None:
    """The "Notes" skip only applies when there's a following `##` to
    prefer; otherwise the generic heading stands."""
    assert extract_title("# Notes\n\nbody with no other heading", "doc.md") == "Notes"


def test_extract_title_ignores_headings_in_non_markdown_extensions() -> None:
    """The heading rules are extension-gated, so a `#` comment in a Python
    file is not mistaken for a title."""
    assert extract_title("# not a markdown heading\n", "script.py") == "script"


def test_expand_braces_single_group() -> None:
    assert _expand_braces("**/*.{py,md,json}") == ["**/*.py", "**/*.md", "**/*.json"]


def test_expand_braces_multiple_groups_yield_cross_product() -> None:
    """Regression: expanding only the first group left the second as a
    literal `{md,py}`, which glob matches against nothing - a collection
    with a two-group pattern silently indexed zero files."""
    assert set(_expand_braces("{src,docs}/**/*.{md,py}")) == {
        "src/**/*.md",
        "src/**/*.py",
        "docs/**/*.md",
        "docs/**/*.py",
    }


def test_expand_braces_nested_groups() -> None:
    assert set(_expand_braces("*.{a,{b,c}}")) == {"*.a", "*.b", "*.c"}


def test_expand_braces_without_group_is_passthrough() -> None:
    assert _expand_braces("**/*.md") == ["**/*.md"]


def test_expand_braces_unbalanced_brace_is_passthrough() -> None:
    """No group to expand, so the pattern reaches glob as the literal
    characters it already was, rather than being mangled."""
    assert _expand_braces("**/*.{md") == ["**/*.{md"]


def test_expand_braces_empty_alternative_is_kept() -> None:
    assert _expand_braces("README{,.md}") == ["README", "README.md"]


def test_expand_braces_dedupes_equivalent_expansions() -> None:
    assert _expand_braces("*.{a,a}") == ["*.a"]


def test_expand_braces_is_capped_for_pathological_patterns() -> None:
    """Ten binary groups is 1024 combinations, each its own filesystem
    walk - the cap stops `update` turning into a hang."""
    assert len(_expand_braces("{a,b}" * 10)) == _MAX_BRACE_EXPANSIONS


def _doc(path: str) -> Document:
    return Document(
        collection_id=1, path=path, title=path, hash="h" * 10,
        created_at=utcnow(), modified_at=utcnow(),
    )


def test_match_named_document_prefers_an_exact_bare_path() -> None:
    refs = [(_doc("notes/alpha.md"), "docs"), (_doc("alpha.md"), "docs")]

    found = _match_named_document(refs, "alpha.md")

    assert found is not None
    assert found[0].path == "alpha.md"


def test_match_named_document_falls_back_to_a_suffix_match() -> None:
    refs = [(_doc("notes/alpha.md"), "docs")]

    found = _match_named_document(refs, "alpha.md")

    assert found is not None
    assert found[0].path == "notes/alpha.md"


def test_match_named_document_virtual_path_requires_the_right_collection() -> None:
    """An exact `marq://` path is resolved against that collection only -
    no suffix fallback - so a wrong collection name finds nothing."""
    refs = [(_doc("alpha.md"), "docs")]

    assert _match_named_document(refs, "marq://other/alpha.md") is None
    assert _match_named_document(refs, "marq://docs/alpha.md") is not None


def test_match_named_document_unknown_path_is_none() -> None:
    assert _match_named_document([(_doc("alpha.md"), "docs")], "nowhere.md") is None


def test_discover_files_matches_every_brace_group(tmp_path: Path) -> None:
    """End-to-end guard on the same bug at the layer that actually feeds
    reindex_collection - no DB needed, just a real directory tree."""
    (tmp_path / "src").mkdir()
    (tmp_path / "docs").mkdir()
    (tmp_path / "src" / "a.py").write_text("a")
    (tmp_path / "src" / "b.md").write_text("b")
    (tmp_path / "docs" / "c.md").write_text("c")
    (tmp_path / "docs" / "d.txt").write_text("d")

    found = _discover_files(str(tmp_path), "{src,docs}/**/*.{md,py}", None)

    assert found == ["docs/c.md", "src/a.py", "src/b.md"]


@pytest.mark.integration
async def test_add_list_rename_remove_collection(session: AsyncSession, user: CurrentUser) -> None:
    await add_collection(session, user, "notes", "/tmp/notes", "**/*.md")
    await session.commit()

    rows = await list_collections(session, user)
    assert [r.name for r in rows] == ["notes"]
    assert rows[0].path == "/tmp/notes"
    assert rows[0].pattern == "**/*.md"

    await rename_collection(session, user, "notes", "renamed")
    await session.commit()
    rows = await list_collections(session, user)
    assert [r.name for r in rows] == ["renamed"]

    result = await remove_collection(session, user, "renamed")
    await session.commit()
    assert result.deleted_docs == 0
    assert await list_collections(session, user) == []


@pytest.mark.integration
async def test_remove_nonexistent_collection_raises(
    session: AsyncSession, user: CurrentUser
) -> None:
    with pytest.raises(CollectionNotFoundError):
        await remove_collection(session, user, "does-not-exist")


@pytest.mark.integration
async def test_context_add_list_remove(session: AsyncSession, user: CurrentUser) -> None:
    await add_collection(session, user, "coll", "/tmp/coll")
    await session.commit()

    await add_context(session, user, "coll", "", "root context")
    await add_context(session, user, "coll", "notes", "notes context")
    await session.commit()

    rows = await list_contexts(session, user)
    assert {(r.path, r.context) for r in rows} == {
        ("", "root context"),
        ("notes", "notes context"),
    }

    removed = await remove_context(session, user, "coll", "notes")
    await session.commit()
    assert removed is True

    removed_again = await remove_context(session, user, "coll", "notes")
    assert removed_again is False


@pytest.mark.integration
async def test_remove_collection_with_context_still_set(
    session: AsyncSession, user: CurrentUser
) -> None:
    """Regression test: removing a collection that still has a per-path
    context row must not raise a raw FK violation - collectioncontext's
    collection_id FK has no ON DELETE CASCADE, so remove_collection must
    delete context rows itself before deleting the collection."""
    await add_collection(session, user, "coll-with-ctx", "/tmp/coll-with-ctx")
    await session.commit()
    await add_context(session, user, "coll-with-ctx", "", "root context")
    await session.commit()

    result = await remove_collection(session, user, "coll-with-ctx")
    await session.commit()
    assert result.deleted_docs == 0
    assert await list_collections(session, user) == []


@pytest.mark.integration
async def test_set_global_context(session: AsyncSession, user: CurrentUser) -> None:
    await set_global_context(session, user, "always include this")
    await session.commit()
    refreshed = await session.get(User, user.id)
    assert refreshed is not None
    assert refreshed.global_context == "always include this"


@pytest.mark.integration
async def test_context_check_flags_collection_with_no_context(
    session: AsyncSession, user: CurrentUser
) -> None:
    collection = await add_collection(session, user, "bare", "/tmp/bare")
    await insert_content(session, "deadbeef", "body")
    await insert_document(
        session, collection.id, "a.md", "A", "deadbeef", utcnow(), utcnow()
    )
    await session.commit()

    missing_collections, missing_paths = await context_check(session, user)
    assert [c.name for c in missing_collections] == ["bare"]
    assert missing_collections[0].doc_count == 1
    assert missing_paths == {}


@pytest.mark.integration
async def test_context_check_flags_uncovered_top_level_dir(
    session: AsyncSession, user: CurrentUser
) -> None:
    collection = await add_collection(session, user, "partial", "/tmp/partial")
    await insert_content(session, "hash1", "body1")
    await insert_document(session, collection.id, "covered/a.md", "A", "hash1", utcnow(), utcnow())
    await insert_content(session, "hash2", "body2")
    await insert_document(
        session, collection.id, "uncovered/b.md", "B", "hash2", utcnow(), utcnow()
    )
    await add_context(session, user, "partial", "covered", "covered dir context")
    await session.commit()

    missing_collections, missing_paths = await context_check(session, user)
    assert missing_collections == []
    assert missing_paths == {"partial": ["uncovered"]}


@pytest.mark.integration
async def test_permission_denied_surfaces_from_can_access(
    session: AsyncSession, user: CurrentUser, monkeypatch: pytest.MonkeyPatch
) -> None:
    await add_collection(session, user, "guarded", "/tmp/guarded")
    await session.commit()

    async def deny(*_args: object, **_kwargs: object) -> bool:
        return False

    monkeypatch.setattr("qmd_py.store.can_access", deny)
    with pytest.raises(PermissionDeniedError):
        await remove_collection(session, user, "guarded")


@pytest.mark.integration
async def test_document_lifecycle_and_content_dedup(
    session: AsyncSession, user: CurrentUser
) -> None:
    collection = await add_collection(session, user, "docs", "/tmp/docs")
    await session.commit()

    hash_a = hash_content("body one")
    await insert_content(session, hash_a, "body one")
    doc = await insert_document(
        session, collection.id, "a.md", "A", hash_a, utcnow(), utcnow()
    )
    await session.commit()

    found = await find_active_document(session, collection.id, "a.md")
    assert found is not None
    assert found.id == doc.id

    hash_b = hash_content("body one revised")
    await insert_content(session, hash_b, "body one revised")
    await update_document(session, found, "A revised", hash_b, utcnow())
    await session.commit()

    cleaned = await cleanup_orphaned_content(session)
    await session.commit()
    assert cleaned == 1  # hash_a is now orphaned

    paths = await get_active_document_paths(session, collection.id)
    assert paths == ["a.md"]

    await deactivate_document(session, collection.id, "a.md")
    await session.commit()
    assert await find_active_document(session, collection.id, "a.md") is None
    assert await get_active_document_paths(session, collection.id) == []
