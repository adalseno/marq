"""Phase 9: MCP server unit tests - the pure/session-taking helpers
behind mcp/server.py's tools and dynamic instructions.

The protocol-level pass (a real MCP ClientSession driving `get`/
`multi_get`/`status` and the `marq://` resource end to end) lives in
test_mcp_server.py. It became possible once conftest.py grew the
`mcp_env` fixture, which points `db.engine.get_session()`'s
process-global engine at a throwaway schema instead of whatever `.env`
resolves to - the obstacle this docstring used to describe. The helpers
below stay unit-tested here because they're worth pinning directly,
without a client session in the way.
"""

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

import qmd_py.cli.commands.mcp as mcp_cmd
from qmd_py.auth import CurrentUser
from qmd_py.cli.commands.mcp import _warn_if_public_bind
from qmd_py.mcp.server import (
    _default_collection_names,
    _encode_qmd_path,
    _filter_by_collections,
    _format_search_summary,
    _is_int,
    _parse_get_lookup,
    _round2,
    build_instructions,
)
from qmd_py.store import add_collection, insert_content, insert_document, utcnow


def test_encode_qmd_path_preserves_slashes_encodes_segments() -> None:
    assert _encode_qmd_path("docs/my file.md") == "docs/my%20file.md"


def test_is_int_rejects_json_booleans() -> None:
    """Python's bool subclasses int, so a bare isinstance(x, int) accepts
    JSON true/false - the fence must not (third review, finding 8)."""
    assert _is_int(10)
    assert _is_int(0)
    assert not _is_int(True)
    assert not _is_int(False)
    assert not _is_int(1.5)
    assert not _is_int("10")


@pytest.mark.parametrize("host", ["127.0.0.1", "127.1.2.3", "localhost", "::1"])
def test_warn_if_public_bind_is_silent_for_loopback(
    host: str, capsys: pytest.CaptureFixture[str]
) -> None:
    _warn_if_public_bind(host)
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.10", "::"])  # noqa: S104
def test_warn_if_public_bind_warns_for_non_loopback(
    host: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """Until real ACL lands, a public bind means unauthenticated access to
    every indexed document - the startup warning is the guard rail against
    doing that by accident (third review, finding 9)."""
    _warn_if_public_bind(host)
    err = capsys.readouterr().err
    assert "Warning" in err
    assert "NO authentication" in err.replace("\n", " ")


def test_start_daemon_keeps_the_previous_stdio_log_as_a_backup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The stdio file is truncated at each start to keep it from growing
    without bound, which wiped a crashed daemon's traceback at the exact
    moment the user retried the start to read it. One backup preserves
    that evidence while still bounding growth at two files."""
    stdio = tmp_path / "mcp-stdio.log"
    backup = tmp_path / "mcp-stdio.log.1"
    stdio.write_text("Traceback (most recent call last): boom")
    monkeypatch.setattr(mcp_cmd, "_STATE_DIR", tmp_path)
    monkeypatch.setattr(mcp_cmd, "_PID_FILE", tmp_path / "mcp.pid")
    monkeypatch.setattr(mcp_cmd, "_STDIO_FILE", stdio)
    monkeypatch.setattr(mcp_cmd, "_STDIO_BACKUP", backup)
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **kw: SimpleNamespace(pid=4242))

    mcp_cmd._start_daemon("127.0.0.1", 8181)

    assert backup.read_text() == "Traceback (most recent call last): boom"
    assert stdio.read_text() == ""
    assert (tmp_path / "mcp.pid").read_text() == "4242"
    assert "pid 4242" in capsys.readouterr().out


def test_round2_rounds_to_two_decimals() -> None:
    assert _round2(0.123456) == 0.12


def test_format_search_summary_empty() -> None:
    assert _format_search_summary([], "foo") == 'No results found for "foo"'


def test_format_search_summary_lists_results() -> None:
    items = [
        {"docid": "#abc123", "file": "a.md", "title": "A", "score": 0.856},
        {"docid": "#def456", "file": "b.md", "title": "B", "score": 0.5},
    ]
    summary = _format_search_summary(items, "foo")
    assert 'Found 2 results for "foo"' in summary
    assert "#abc123 86% a.md - A" in summary
    assert "#def456 50% b.md - B" in summary


def test_filter_by_collections_noop_for_single_collection() -> None:
    class R:
        def __init__(self, file: str) -> None:
            self.file = file

    results = [R("marq://a/x.md")]
    assert _filter_by_collections(results, ["a"]) == results


def test_filter_by_collections_empty_scope_drops_everything() -> None:
    """Regression: an empty name list means nothing is in default scope
    (every collection excluded), not "no filter". Treating it as a no-op
    made `collection exclude` return everything instead of nothing."""

    class R:
        def __init__(self, file: str) -> None:
            self.file = file

    assert _filter_by_collections([R("marq://a/x.md")], []) == []


def test_filter_by_collections_filters_for_multiple() -> None:
    class R:
        def __init__(self, file: str) -> None:
            self.file = file

    results = [R("marq://a/x.md"), R("marq://b/y.md"), R("marq://c/z.md")]
    filtered = _filter_by_collections(results, ["a", "b"])
    assert [r.file for r in filtered] == ["marq://a/x.md", "marq://b/y.md"]


@pytest.mark.parametrize(
    ("file", "expected_lookup", "expected_from", "expected_max"),
    [
        ("foo.md", "foo.md", None, None),
        ("foo.md:100", "foo.md", 100, None),
        ("foo.md:100:40", "foo.md", 100, 40),
        ("#abc123:5:10", "#abc123", 5, 10),
    ],
)
def test_parse_get_lookup(
    file: str, expected_lookup: str, expected_from: int | None, expected_max: int | None
) -> None:
    lookup, from_line, max_lines = _parse_get_lookup(file, None, None)
    assert lookup == expected_lookup
    assert from_line == expected_from
    assert max_lines == expected_max


def test_parse_get_lookup_explicit_args_take_precedence() -> None:
    lookup, from_line, max_lines = _parse_get_lookup("foo.md:100:40", 5, 6)
    assert lookup == "foo.md"
    assert from_line == 5
    assert max_lines == 6


def test_parse_get_lookup_clamps_from_line_to_one() -> None:
    _, from_line, _ = _parse_get_lookup("foo.md", -3, None)
    assert from_line == 1


@pytest.mark.integration
async def test_default_collection_names_only_include_by_default(
    session: AsyncSession, user: CurrentUser
) -> None:
    await add_collection(session, user, "included", "/tmp/included")
    excluded = await add_collection(session, user, "excluded", "/tmp/excluded")
    excluded.include_by_default = False
    session.add(excluded)
    await session.commit()

    names = await _default_collection_names(session, user)
    assert names == ["included"]


@pytest.mark.integration
async def test_build_instructions_mentions_doc_count_and_collections(
    session: AsyncSession, user: CurrentUser
) -> None:
    collection = await add_collection(session, user, "notes", "/tmp/notes")
    await insert_content(session, "hash1", "hello world")
    await insert_document(session, collection.id, "a.md", "A", "hash1", utcnow(), utcnow())
    await session.commit()

    instructions = await build_instructions(session, user, "bge-m3-q8_0")
    assert "1 documents" in instructions
    assert "notes" in instructions
    assert "No vector embeddings yet" in instructions
    assert "Search: Use `query`" in instructions
