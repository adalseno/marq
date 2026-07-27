"""Phase 11: formatter.py unit tests. Pure/stateless (no DB/session
dependency - only SearchResult/MultiGetFile/DocumentDetail dataclass
instances), so unlike the CLI commands that call these functions, this
module is fully pytestable in isolation."""

import json
from datetime import UTC, datetime

from qmd_py.cli.formatter import (
    FormatOptions,
    document_to_json,
    document_to_markdown,
    document_to_xml,
    documents_to_csv,
    documents_to_files,
    documents_to_json,
    documents_to_markdown,
    documents_to_toon,
    documents_to_xml,
    escape_csv,
    escape_xml,
    format_document,
    format_documents,
    format_search_results,
    search_results_to_csv,
    search_results_to_files,
    search_results_to_json,
    search_results_to_markdown,
    search_results_to_toon,
    search_results_to_xml,
)
from qmd_py.search.fts import SearchResult
from qmd_py.store import DocumentDetail, MultiGetFile

_NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _result(**overrides: object) -> SearchResult:
    defaults: dict[str, object] = {
        "filepath": "marq://notes/a.md",
        "display_path": "notes/a.md",
        "title": "A Doc",
        "hash": "abc123def456",
        "docid": "abc123",
        "collection_name": "notes",
        "modified_at": _NOW,
        "body_length": 20,
        "body": "hello world, this is the body",
        "context": None,
        "score": 0.856,
    }
    defaults.update(overrides)
    return SearchResult(**defaults)  # type: ignore[arg-type]


def _multi_get_file(**overrides: object) -> MultiGetFile:
    defaults: dict[str, object] = {
        "filepath": "marq://notes/a.md",
        "display_path": "notes/a.md",
        "title": "A Doc",
        "body": "hello world",
        "context": None,
        "skipped": False,
        "docid": "abc123",
        "skip_reason": None,
    }
    defaults.update(overrides)
    return MultiGetFile(**defaults)  # type: ignore[arg-type]


def _document_detail(**overrides: object) -> DocumentDetail:
    defaults: dict[str, object] = {
        "filepath": "marq://notes/a.md",
        "display_path": "notes/a.md",
        "title": "A Doc",
        "context": None,
        "hash": "abc123def456",
        "docid": "abc123",
        "collection_name": "notes",
        "modified_at": _NOW,
        "body_length": 11,
        "body": "hello world",
    }
    defaults.update(overrides)
    return DocumentDetail(**defaults)  # type: ignore[arg-type]


# =============================================================================
# escape helpers
# =============================================================================


def test_escape_csv_plain_value_unchanged() -> None:
    assert escape_csv("plain") == "plain"


def test_escape_csv_quotes_value_with_comma() -> None:
    assert escape_csv("a,b") == '"a,b"'


def test_escape_csv_doubles_embedded_quotes() -> None:
    assert escape_csv('he said "hi"') == '"he said ""hi"""'


def test_escape_csv_none_is_empty() -> None:
    assert escape_csv(None) == ""


def test_escape_xml_escapes_all_entities() -> None:
    assert escape_xml("<a> & 'b' \"c\"") == "&lt;a&gt; &amp; &apos;b&apos; &quot;c&quot;"


# =============================================================================
# Search results - all formats
# =============================================================================


def test_search_results_to_json_shape() -> None:
    results = [_result()]
    entries = json.loads(search_results_to_json(results, FormatOptions(query="hello")))
    assert entries[0]["docid"] == "#abc123"
    assert entries[0]["file"] == "notes/a.md"
    assert entries[0]["title"] == "A Doc"
    assert "context" not in entries[0]
    assert "snippet" in entries[0]
    assert "body" not in entries[0]


def test_search_results_to_json_full_includes_body_not_snippet() -> None:
    results = [_result()]
    entries = json.loads(search_results_to_json(results, FormatOptions(query="hello", full=True)))
    assert entries[0]["body"] == results[0].body
    assert "snippet" not in entries[0]


def test_search_results_to_json_includes_context_when_present() -> None:
    results = [_result(context="some folder context")]
    entries = json.loads(search_results_to_json(results, FormatOptions(query="hello")))
    assert entries[0]["context"] == "some folder context"


def test_search_results_to_json_empty_list() -> None:
    assert json.loads(search_results_to_json([], FormatOptions())) == []


def test_search_results_to_json_line_numbers_prefix_body_and_snippet() -> None:
    results = [_result()]
    entries = json.loads(
        search_results_to_json(results, FormatOptions(query="hello", full=True, line_numbers=True))
    )
    assert entries[0]["body"].startswith("1: ")

    entries = json.loads(
        search_results_to_json(results, FormatOptions(query="hello", line_numbers=True))
    )
    assert entries[0]["snippet"].startswith("1: ")


def test_search_results_to_csv_header_and_row_count() -> None:
    # A single result's body/snippet may itself contain embedded newlines
    # once quoted, so count rows by a distinguishing field rather than by
    # splitting on "\n".
    results = [_result(), _result(display_path="notes/b.md")]
    csv_out = search_results_to_csv(results, FormatOptions())
    assert csv_out.startswith("docid,score,file,title,context,line,snippet\n")
    assert csv_out.count("notes/a.md") == 1
    assert csv_out.count("notes/b.md") == 1


def test_search_results_to_csv_line_numbers_prefixes_content() -> None:
    csv_out = search_results_to_csv([_result()], FormatOptions(query="hello", line_numbers=True))
    assert '"1: ' in csv_out


def test_search_results_to_files_format() -> None:
    out = search_results_to_files([_result()])
    assert out == "#abc123,0.86,notes/a.md"


def test_search_results_to_files_includes_quoted_context() -> None:
    out = search_results_to_files([_result(context='has "quotes"')])
    assert out == '#abc123,0.86,notes/a.md,"has ""quotes"""'


def test_search_results_to_xml_wraps_each_file() -> None:
    out = search_results_to_xml([_result()], FormatOptions(query="hello"))
    assert out.startswith('<file docid="#abc123" name="notes/a.md" title="A Doc">')
    assert out.endswith("</file>")


def test_search_results_to_xml_line_numbers_prefixes_content() -> None:
    out = search_results_to_xml([_result()], FormatOptions(query="hello", line_numbers=True))
    assert "\n1: " in out


def test_search_results_to_markdown_full_shows_body_not_snippet() -> None:
    out = search_results_to_markdown([_result()], FormatOptions(query="hello", full=True))
    assert "hello world, this is the body" in out


def test_search_results_to_markdown_line_numbers_prefixes_content() -> None:
    out = search_results_to_markdown([_result()], FormatOptions(query="hello", line_numbers=True))
    assert "1: " in out


def test_search_results_to_toon_header_and_row() -> None:
    out = search_results_to_toon([_result()], FormatOptions(query="hello"))
    lines = out.split("\n")
    assert lines[0] == "results[1]{docid,score,file,title,context,line,snippet}:"
    assert len(lines) == 2


def test_search_results_to_toon_quotes_value_containing_comma() -> None:
    out = search_results_to_toon([_result(title="A, B")], FormatOptions(query="hello"))
    assert '"A, B"' in out


def test_search_results_to_toon_line_numbers_prefixes_content() -> None:
    out = search_results_to_toon([_result()], FormatOptions(query="hello", line_numbers=True))
    assert "1: " in out


def test_format_search_results_dispatches_by_format() -> None:
    results = [_result()]
    assert format_search_results(results, "json").startswith("[")
    assert format_search_results(results, "files").startswith("#abc123")
    assert format_search_results(results, "xml").startswith("<file")
    assert format_search_results(results, "toon").startswith("results[")
    assert format_search_results(results, "csv").startswith("docid,score")


def test_format_search_results_cli_falls_back_to_markdown() -> None:
    cli_out = format_search_results([_result()], "cli")
    md_out = format_search_results([_result()], "md")
    assert cli_out == md_out
    assert "# A Doc" in cli_out


# =============================================================================
# Multi-get documents - all formats
# =============================================================================


def test_documents_to_json_includes_body_when_not_skipped() -> None:
    entries = json.loads(documents_to_json([_multi_get_file()]))
    assert entries[0]["docid"] == "#abc123"
    assert entries[0]["body"] == "hello world"
    assert "skipped" not in entries[0]


def test_documents_to_json_includes_context_when_present() -> None:
    entries = json.loads(documents_to_json([_multi_get_file(context="folder note")]))
    assert entries[0]["context"] == "folder note"


def test_documents_to_json_skipped_omits_body_and_flags_reason() -> None:
    skipped = _multi_get_file(skipped=True, skip_reason="too large", docid=None)
    entries = json.loads(documents_to_json([skipped]))
    assert entries[0]["skipped"] is True
    assert entries[0]["reason"] == "too large"
    assert "docid" not in entries[0]
    assert "body" not in entries[0]


def test_documents_to_csv_shape() -> None:
    csv_out = documents_to_csv([_multi_get_file()])
    lines = csv_out.split("\n")
    assert lines[0] == "docid,file,title,context,skipped,body"
    assert lines[1] == "#abc123,notes/a.md,A Doc,,false,hello world"


def test_documents_to_files_marks_skipped() -> None:
    out = documents_to_files([_multi_get_file(skipped=True, skip_reason="too large")])
    assert out == "#abc123 notes/a.md,[SKIPPED]"


def test_documents_to_markdown_omits_duplicate_title() -> None:
    same_title = _multi_get_file(title="notes/a.md")
    out = documents_to_markdown([same_title])
    assert "**Title:**" not in out


def test_documents_to_markdown_includes_distinct_title() -> None:
    out = documents_to_markdown([_multi_get_file(title="A Doc")])
    assert "**Title:** A Doc" in out


def test_documents_to_markdown_includes_context() -> None:
    out = documents_to_markdown([_multi_get_file(context="folder note")])
    assert "**Context:** folder note" in out


def test_documents_to_markdown_shows_skip_reason() -> None:
    out = documents_to_markdown([_multi_get_file(skipped=True, skip_reason="too large")])
    assert "> too large" in out
    assert "```" not in out


def test_documents_to_xml_wraps_in_documents_root() -> None:
    out = documents_to_xml([_multi_get_file()])
    assert out.startswith('<?xml version="1.0" encoding="UTF-8"?>\n<documents>')
    assert out.endswith("</documents>")
    assert "<file>notes/a.md</file>" in out


def test_documents_to_xml_includes_context() -> None:
    out = documents_to_xml([_multi_get_file(context="folder note")])
    assert "<context>folder note</context>" in out


def test_documents_to_xml_skipped_includes_reason_not_body() -> None:
    out = documents_to_xml([_multi_get_file(skipped=True, skip_reason="too large")])
    assert "<skipped>true</skipped>" in out
    assert "<reason>too large</reason>" in out
    assert "<body>" not in out


def test_documents_to_toon_row_shape() -> None:
    out = documents_to_toon([_multi_get_file()])
    lines = out.split("\n")
    assert lines[0] == "documents[1]{docid,file,title,context,skipped,body}:"


def test_format_documents_dispatches_by_format() -> None:
    docs = [_multi_get_file()]
    assert format_documents(docs, "json").startswith("[")
    assert format_documents(docs, "csv").startswith("docid,file")
    assert format_documents(docs, "files") == "#abc123 notes/a.md"
    assert format_documents(docs, "xml").startswith("<?xml")
    assert format_documents(docs, "toon").startswith("documents[")
    assert "## notes/a.md" in format_documents(docs, "cli")


# =============================================================================
# Single document
# =============================================================================


def test_document_to_json_shape() -> None:
    entry = json.loads(document_to_json(_document_detail()))
    assert entry["file"] == "notes/a.md"
    assert entry["hash"] == "abc123def456"
    assert entry["bodyLength"] == 11
    assert entry["body"] == "hello world"
    assert "context" not in entry


def test_document_to_json_includes_context_when_present() -> None:
    entry = json.loads(document_to_json(_document_detail(context="folder note")))
    assert entry["context"] == "folder note"


def test_document_to_markdown_shape() -> None:
    out = document_to_markdown(_document_detail())
    assert out.startswith("# A Doc\n")
    assert "**File:** notes/a.md" in out
    assert "hello world" in out


def test_document_to_markdown_includes_context() -> None:
    out = document_to_markdown(_document_detail(context="folder note"))
    assert "**Context:** folder note" in out


def test_document_to_xml_shape() -> None:
    out = document_to_xml(_document_detail())
    assert "<file>notes/a.md</file>" in out
    assert "<body>hello world</body>" in out


def test_document_to_xml_includes_context() -> None:
    out = document_to_xml(_document_detail(context="folder note"))
    assert "<context>folder note</context>" in out


def test_format_document_dispatches_by_format() -> None:
    doc = _document_detail()
    assert format_document(doc, "json").startswith("{")
    assert format_document(doc, "xml").startswith("<?xml")
    assert format_document(doc, "md").startswith("# A Doc")
    # Anything else (including "cli") falls back to markdown.
    assert format_document(doc, "cli") == format_document(doc, "md")
