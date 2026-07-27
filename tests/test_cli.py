"""CLI command tests, driven through click's `CliRunner` against a real
throwaway Postgres schema (see conftest.py's `marq` fixture).

These exercise the command bodies end to end - argument parsing, the
service-layer call, output formatting, and exit codes - which unit tests
of `store.py` alone never reach. Commands needing the LLM router
(`vsearch`, `query`, `embed`, `doctor`, `bench`) are covered separately;
everything here runs against Postgres only.

Tests are sync on purpose: each command body ends in `asyncio.run()`,
which refuses to start inside an already-running loop.
"""

import json
from collections.abc import Callable
from pathlib import Path

import pytest
from click.testing import Result

pytestmark = pytest.mark.integration

Marq = Callable[..., Result]


def _write_collection(tmp_path: Path) -> Path:
    """A small on-disk collection: two markdown files and one ignored."""
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "alpha.md").write_text("# Alpha\n\nunique-alpha-token here\n")
    (tmp_path / "notes" / "beta.md").write_text("# Beta\n\nunique-beta-token there\n")
    (tmp_path / "notes" / "ignored.txt").write_text("not indexed\n")
    return tmp_path


def test_collection_add_indexes_files(marq: Marq, tmp_path: Path) -> None:
    _write_collection(tmp_path)

    result = marq("collection", "add", str(tmp_path), "--name", "docs", "--mask", "**/*.md")

    assert result.exit_code == 0, result.output
    assert "Indexed: 2 new" in result.output


def test_collection_add_rejects_duplicate_name(marq: Marq, tmp_path: Path) -> None:
    _write_collection(tmp_path)
    marq("collection", "add", str(tmp_path), "--name", "docs", "--mask", "**/*.md")

    result = marq("collection", "add", str(tmp_path), "--name", "docs", "--mask", "**/*.md")

    assert result.exit_code == 1
    assert "already exists" in result.output


def test_ls_without_argument_lists_collections(marq: Marq, tmp_path: Path) -> None:
    _write_collection(tmp_path)
    marq("collection", "add", str(tmp_path), "--name", "docs", "--mask", "**/*.md")

    result = marq("ls")

    assert result.exit_code == 0, result.output
    assert "marq://docs/  (2 files)" in result.output


def test_ls_on_empty_index_suggests_next_step(marq: Marq) -> None:
    result = marq("ls")

    assert result.exit_code == 0, result.output
    assert "No collections found" in result.output


def test_ls_scoped_to_collection_lists_files(marq: Marq, tmp_path: Path) -> None:
    _write_collection(tmp_path)
    marq("collection", "add", str(tmp_path), "--name", "docs", "--mask", "**/*.md")

    result = marq("ls", "docs")

    assert result.exit_code == 0, result.output
    assert "marq://docs/notes/alpha.md" in result.output
    assert "marq://docs/notes/beta.md" in result.output
    assert "ignored.txt" not in result.output


def test_ls_unknown_collection_exits_nonzero(marq: Marq) -> None:
    result = marq("ls", "nope")

    assert result.exit_code == 1
    assert "Collection not found: nope" in result.output


def test_search_finds_indexed_document(marq: Marq, tmp_path: Path) -> None:
    _write_collection(tmp_path)
    marq("collection", "add", str(tmp_path), "--name", "docs", "--mask", "**/*.md")

    result = marq("search", "unique-alpha-token")

    assert result.exit_code == 0, result.output
    assert "alpha.md" in result.output
    assert "beta.md" not in result.output


def test_search_reports_no_results(marq: Marq, tmp_path: Path) -> None:
    _write_collection(tmp_path)
    marq("collection", "add", str(tmp_path), "--name", "docs", "--mask", "**/*.md")

    result = marq("search", "term-that-appears-nowhere")

    assert result.exit_code == 0, result.output
    assert "No results found." in result.output


def test_search_json_format_is_machine_readable(marq: Marq, tmp_path: Path) -> None:
    """`file` is the collection-prefixed display path, not the `marq://`
    virtual path - the same field name every other --format uses."""
    _write_collection(tmp_path)
    marq("collection", "add", str(tmp_path), "--name", "docs", "--mask", "**/*.md")

    result = marq("search", "unique-alpha-token", "--format", "json")

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert [r["file"] for r in payload] == ["docs/notes/alpha.md"]


def test_get_by_path_prints_body_with_line_numbers(marq: Marq, tmp_path: Path) -> None:
    _write_collection(tmp_path)
    marq("collection", "add", str(tmp_path), "--name", "docs", "--mask", "**/*.md")

    result = marq("get", "notes/alpha.md")

    assert result.exit_code == 0, result.output
    assert "marq://docs/notes/alpha.md" in result.output
    assert "1: # Alpha" in result.output


def test_get_missing_document_exits_nonzero(marq: Marq, tmp_path: Path) -> None:
    _write_collection(tmp_path)
    marq("collection", "add", str(tmp_path), "--name", "docs", "--mask", "**/*.md")

    result = marq("get", "notes/nonexistent.md")

    assert result.exit_code == 1
    assert "Document not found" in result.output


def test_get_honours_line_range_suffix(marq: Marq, tmp_path: Path) -> None:
    (tmp_path / "many.md").write_text("# T\n" + "\n".join(f"line{i}" for i in range(1, 21)))
    marq("collection", "add", str(tmp_path), "--name", "docs", "--mask", "**/*.md")

    result = marq("get", "many.md:3:2")

    assert result.exit_code == 0, result.output
    assert "3: line2" in result.output
    assert "4: line3" in result.output
    assert "line5" not in result.output


def test_multi_get_matches_glob(marq: Marq, tmp_path: Path) -> None:
    _write_collection(tmp_path)
    marq("collection", "add", str(tmp_path), "--name", "docs", "--mask", "**/*.md")

    result = marq("multi-get", "notes/*.md")

    assert result.exit_code == 0, result.output
    assert "alpha.md" in result.output
    assert "beta.md" in result.output


def test_multi_get_without_match_exits_nonzero(marq: Marq, tmp_path: Path) -> None:
    _write_collection(tmp_path)
    marq("collection", "add", str(tmp_path), "--name", "docs", "--mask", "**/*.md")

    result = marq("multi-get", "nothing/*.md")

    assert result.exit_code == 1
    assert "No files matched pattern" in result.output


def test_status_reports_document_count(marq: Marq, tmp_path: Path) -> None:
    _write_collection(tmp_path)
    marq("collection", "add", str(tmp_path), "--name", "docs", "--mask", "**/*.md")

    result = marq("status")

    assert result.exit_code == 0, result.output
    assert "Total: 2 files indexed" in result.output
    assert "docs (marq://docs/)" in result.output


def test_collection_list_show_rename_remove_roundtrip(marq: Marq, tmp_path: Path) -> None:
    _write_collection(tmp_path)
    marq("collection", "add", str(tmp_path), "--name", "docs", "--mask", "**/*.md")

    listed = marq("collection", "list")
    assert "docs (marq://docs/)" in listed.output

    shown = marq("collection", "show", "docs")
    assert str(tmp_path) in shown.output
    assert "**/*.md" in shown.output

    renamed = marq("collection", "rename", "docs", "notes")
    assert renamed.exit_code == 0, renamed.output

    removed = marq("collection", "remove", "notes")
    assert removed.exit_code == 0, removed.output
    assert "Deleted 2 documents" in removed.output
    assert "No collections found" in marq("collection", "list").output


def test_collection_remove_unknown_exits_nonzero(marq: Marq) -> None:
    result = marq("collection", "remove", "ghost")

    assert result.exit_code == 1
    assert "Collection not found: ghost" in result.output


def test_collection_exclude_include_roundtrip(marq: Marq, tmp_path: Path) -> None:
    _write_collection(tmp_path)
    marq("collection", "add", str(tmp_path), "--name", "docs", "--mask", "**/*.md")

    excluded = marq("collection", "exclude", "docs")
    assert excluded.exit_code == 0, excluded.output
    assert "[excluded]" in marq("collection", "list").output

    included = marq("collection", "include", "docs")
    assert included.exit_code == 0, included.output
    assert "[excluded]" not in marq("collection", "list").output


def test_excluded_collection_still_searched_when_scoped_explicitly(
    marq: Marq, tmp_path: Path
) -> None:
    """`-c` is an explicit override: it wins over include_by_default."""
    _write_collection(tmp_path)
    marq("collection", "add", str(tmp_path), "--name", "docs", "--mask", "**/*.md")
    marq("collection", "exclude", "docs")

    result = marq("search", "unique-alpha-token", "-c", "docs")

    assert result.exit_code == 0, result.output
    assert "alpha.md" in result.output


def test_excluding_every_collection_empties_default_search_scope(
    marq: Marq, tmp_path: Path
) -> None:
    _write_collection(tmp_path)
    marq("collection", "add", str(tmp_path), "--name", "docs", "--mask", "**/*.md")
    assert "alpha.md" in marq("search", "unique-alpha-token").output

    marq("collection", "exclude", "docs")

    assert "No results found." in marq("search", "unique-alpha-token").output


def test_context_add_list_remove(marq: Marq, tmp_path: Path) -> None:
    _write_collection(tmp_path)
    marq("collection", "add", str(tmp_path), "--name", "docs", "--mask", "**/*.md")

    added = marq("context", "add", "docs/notes", "these are my notes")
    assert added.exit_code == 0, added.output

    listed = marq("context", "list")
    assert "these are my notes" in listed.output

    got = marq("get", "notes/alpha.md")
    assert "these are my notes" in got.output

    removed = marq("context", "rm", "docs/notes")
    assert removed.exit_code == 0, removed.output
    assert "these are my notes" not in marq("context", "list").output


def test_context_check_flags_collection_without_context(marq: Marq, tmp_path: Path) -> None:
    _write_collection(tmp_path)
    marq("collection", "add", str(tmp_path), "--name", "docs", "--mask", "**/*.md")

    result = marq("context", "check")

    assert result.exit_code == 0, result.output
    assert "docs" in result.output


def test_update_reindexes_changed_files(marq: Marq, tmp_path: Path) -> None:
    _write_collection(tmp_path)
    marq("collection", "add", str(tmp_path), "--name", "docs", "--mask", "**/*.md")

    (tmp_path / "notes" / "gamma.md").write_text("# Gamma\n\nunique-gamma-token\n")
    result = marq("update")

    assert result.exit_code == 0, result.output
    assert "gamma.md" in marq("search", "unique-gamma-token").output


def test_cleanup_runs_and_reports(marq: Marq, tmp_path: Path) -> None:
    _write_collection(tmp_path)
    marq("collection", "add", str(tmp_path), "--name", "docs", "--mask", "**/*.md")
    (tmp_path / "notes" / "beta.md").unlink()
    marq("update")

    result = marq("cleanup")

    assert result.exit_code == 0, result.output
