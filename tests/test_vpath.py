"""Phase 11: vpath.py unit tests (pure, no DB)."""

from qmd_py.vpath import build_virtual_path, is_virtual_path, parse_virtual_path


def test_is_virtual_path_qmd_scheme() -> None:
    assert is_virtual_path("marq://notes/a.md")


def test_is_virtual_path_bare_double_slash() -> None:
    assert is_virtual_path("//notes/a.md")


def test_is_virtual_path_false_for_plain_path() -> None:
    assert not is_virtual_path("notes/a.md")


def test_parse_virtual_path_splits_collection_and_path() -> None:
    assert parse_virtual_path("marq://notes/sub/a.md") == ("notes", "sub/a.md")


def test_parse_virtual_path_collection_root() -> None:
    assert parse_virtual_path("marq://notes/") == ("notes", "")
    assert parse_virtual_path("marq://notes") == ("notes", "")


def test_parse_virtual_path_rejects_non_virtual() -> None:
    assert parse_virtual_path("notes/a.md") is None


def test_build_virtual_path_joins_collection_and_path() -> None:
    assert build_virtual_path("notes", "sub/a.md") == "marq://notes/sub/a.md"
