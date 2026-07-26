"""Output formatting - port of the TS reference's `formatter.ts` plus the
CLI's own inline multi-get formatting (src/cli/qmd.ts), which includes a
`docid` field formatter.ts's own library-level `documentsToJson`/etc.
don't - the CLI's actual `multi-get` output is what's frozen here, not
the library function.

Six formats port the TS reference faithfully: `json`, `csv`, `md`, `xml`,
`files`, and `cli` (the human-readable default, which falls back to `md`
- same as the TS reference: "CLI format should be handled separately with
colors... return a simple text version as fallback"). A 7th, `toon`
(https://toonformat.dev/), is a qmd-py-only addition - not in the TS
reference - since qmd is fundamentally an LLM-facing search tool and
TOON's whole purpose is cutting prompt-token overhead for exactly this
shape of data (a uniform array of small records). It reuses the same
fixed field layout as `csv` (empty string for a row's absent optional
fields) rather than per-row field lists, since TOON's tabular form
requires one declared field list for the whole array.
"""

import json
import re
from dataclasses import dataclass, field

from qmd_py.cli.snippet import extract_snippet
from qmd_py.search.fts import SearchResult
from qmd_py.store import DocumentDetail, MultiGetFile, add_line_numbers

OutputFormat = str  # "cli" | "json" | "csv" | "md" | "xml" | "files" | "toon"


def escape_csv(value: str | int | float | None) -> str:
    if value is None:
        return ""
    text = str(value)
    if "," in text or '"' in text or "\n" in text:
        return '"' + text.replace('"', '""') + '"'
    return text


def escape_xml(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


_TOON_SPECIAL_CHARS = re.compile(r'[:"\\\[\]{}\x00-\x1f]')


def _toon_scalar(value: object) -> str:
    """One TOON scalar value, quoted per the spec's rules: empty, boolean/
    null-lookalike, numeric-lookalike, or containing a delimiter/structural
    character all require quotes."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return str(value)
    text = "" if value is None else str(value)
    looks_numeric = True
    try:
        float(text)
    except ValueError:
        looks_numeric = False
    needs_quote = (
        text == ""
        or text != text.strip()
        or text in ("true", "false", "null")
        or looks_numeric
        or "," in text
        or bool(_TOON_SPECIAL_CHARS.search(text))
    )
    if not needs_quote:
        return text
    escaped = text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    return f'"{escaped}"'


def _to_toon_table(key: str, field_names: list[str], rows: list[list[object]]) -> str:
    header = f"{key}[{len(rows)}]{{{','.join(field_names)}}}:"
    lines = [header]
    lines.extend("  " + ",".join(_toon_scalar(v) for v in row) for row in rows)
    return "\n".join(lines)


@dataclass
class FormatOptions:
    full: bool = False
    query: str = ""
    line_numbers: bool = False
    intent: str | None = field(default=None)


def _content_or_snippet(row: SearchResult, opts: FormatOptions, max_len: int) -> str:
    if opts.full:
        return row.body
    return extract_snippet(row.body, opts.query, max_len, row.chunk_pos, None, opts.intent).snippet


# =============================================================================
# Search results
# =============================================================================


def search_results_to_json(results: list[SearchResult], opts: FormatOptions) -> str:
    entries: list[dict[str, object]] = []
    for row in results:
        snippet_info = (
            extract_snippet(row.body, opts.query, 300, row.chunk_pos, None, opts.intent)
            if row.body
            else None
        )
        body = row.body if opts.full else None
        snippet = snippet_info.snippet if (snippet_info and not opts.full) else None

        if opts.line_numbers:
            if body:
                body = add_line_numbers(body)
            if snippet:
                snippet = add_line_numbers(snippet)

        entry: dict[str, object] = {
            "docid": f"#{row.docid}",
            "score": round(row.score, 2),
            "file": row.display_path,
        }
        if snippet_info:
            entry["line"] = snippet_info.line
        entry["title"] = row.title
        if row.context:
            entry["context"] = row.context
        if body:
            entry["body"] = body
        if snippet:
            entry["snippet"] = snippet
        entries.append(entry)
    return json.dumps(entries, indent=2)


def search_results_to_csv(results: list[SearchResult], opts: FormatOptions) -> str:
    rows = ["docid,score,file,title,context,line,snippet"]
    for row in results:
        snippet_info = extract_snippet(row.body, opts.query, 500, row.chunk_pos, None, opts.intent)
        content = row.body if opts.full else snippet_info.snippet
        if opts.line_numbers and content:
            content = add_line_numbers(content)
        rows.append(
            ",".join(
                [
                    f"#{row.docid}",
                    f"{row.score:.4f}",
                    escape_csv(row.display_path),
                    escape_csv(row.title),
                    escape_csv(row.context or ""),
                    str(snippet_info.line),
                    escape_csv(content),
                ]
            )
        )
    return "\n".join(rows)


def search_results_to_files(results: list[SearchResult]) -> str:
    lines = []
    for row in results:
        ctx = f',"{row.context.replace(chr(34), chr(34) * 2)}"' if row.context else ""
        lines.append(f"#{row.docid},{row.score:.2f},{row.display_path}{ctx}")
    return "\n".join(lines)


def search_results_to_markdown(results: list[SearchResult], opts: FormatOptions) -> str:
    blocks = []
    for row in results:
        heading = row.title or row.display_path
        content = _content_or_snippet(row, opts, 500)
        if opts.line_numbers:
            content = add_line_numbers(content)
        file_line = f"**file:** `{row.display_path}`\n"
        context_line = f"**context:** {row.context}\n" if row.context else ""
        blocks.append(
            f"---\n# {heading}\n\n{file_line}**docid:** `#{row.docid}`\n{context_line}\n{content}\n"
        )
    return "\n".join(blocks)


def search_results_to_xml(results: list[SearchResult], opts: FormatOptions) -> str:
    items = []
    for row in results:
        title_attr = f' title="{escape_xml(row.title)}"' if row.title else ""
        content = _content_or_snippet(row, opts, 500)
        if opts.line_numbers:
            content = add_line_numbers(content)
        context_attr = f' context="{escape_xml(row.context)}"' if row.context else ""
        items.append(
            f'<file docid="#{row.docid}" name="{escape_xml(row.display_path)}"'
            f"{title_attr}{context_attr}>\n{escape_xml(content)}\n</file>"
        )
    return "\n\n".join(items)


def search_results_to_toon(results: list[SearchResult], opts: FormatOptions) -> str:
    field_names = ["docid", "score", "file", "title", "context", "line", "snippet"]
    rows: list[list[object]] = []
    for row in results:
        snippet_info = (
            extract_snippet(row.body, opts.query, 300, row.chunk_pos, None, opts.intent)
            if row.body
            else None
        )
        content = row.body if opts.full else (snippet_info.snippet if snippet_info else "")
        if opts.line_numbers and content:
            content = add_line_numbers(content)
        rows.append(
            [
                f"#{row.docid}",
                round(row.score, 2),
                row.display_path,
                row.title,
                row.context or "",
                snippet_info.line if snippet_info else "",
                content,
            ]
        )
    return _to_toon_table("results", field_names, rows)


def format_search_results(
    results: list[SearchResult], out_format: OutputFormat, opts: FormatOptions | None = None
) -> str:
    opts = opts or FormatOptions()
    if out_format == "json":
        return search_results_to_json(results, opts)
    if out_format == "csv":
        return search_results_to_csv(results, opts)
    if out_format == "files":
        return search_results_to_files(results)
    if out_format == "xml":
        return search_results_to_xml(results, opts)
    if out_format == "toon":
        return search_results_to_toon(results, opts)
    # "cli" falls back to markdown, same as the TS reference.
    return search_results_to_markdown(results, opts)


# =============================================================================
# Multi-get documents - mirrors the CLI's own inline formatting in
# qmd.ts's multiGet(), which (unlike formatter.ts's library-level
# documentsToJson/etc.) always includes docid.
# =============================================================================


def documents_to_json(results: list[MultiGetFile]) -> str:
    entries: list[dict[str, object]] = []
    for r in results:
        entry: dict[str, object] = {"file": r.display_path}
        if r.docid:
            entry["docid"] = f"#{r.docid}"
        entry["title"] = r.title
        if r.context:
            entry["context"] = r.context
        if r.skipped:
            entry["skipped"] = True
            entry["reason"] = r.skip_reason
        else:
            entry["body"] = r.body
        entries.append(entry)
    return json.dumps(entries, indent=2)


def documents_to_csv(results: list[MultiGetFile]) -> str:
    rows = ["docid,file,title,context,skipped,body"]
    for r in results:
        docid_val = f"#{r.docid}" if r.docid else ""
        rows.append(
            ",".join(
                escape_csv(v)
                for v in (
                    docid_val,
                    r.display_path,
                    r.title,
                    r.context or "",
                    "true" if r.skipped else "false",
                    r.skip_reason if r.skipped else r.body,
                )
            )
        )
    return "\n".join(rows)


def documents_to_files(results: list[MultiGetFile]) -> str:
    lines = []
    for r in results:
        id_prefix = f"#{r.docid} " if r.docid else ""
        ctx = f',"{r.context.replace(chr(34), chr(34) * 2)}"' if r.context else ""
        status = ",[SKIPPED]" if r.skipped else ""
        lines.append(f"{id_prefix}{r.display_path}{ctx}{status}")
    return "\n".join(lines)


def documents_to_markdown(results: list[MultiGetFile]) -> str:
    blocks = []
    for r in results:
        parts = [f"## {r.display_path}\n"]
        if r.docid:
            parts.append(f"**docid:** `#{r.docid}`\n")
        if r.title and r.title != r.display_path:
            parts.append(f"**Title:** {r.title}\n")
        if r.context:
            parts.append(f"**Context:** {r.context}\n")
        if r.skipped:
            parts.append(f"> {r.skip_reason}\n")
        else:
            parts.append("```")
            parts.append(r.body)
            parts.append("```\n")
        blocks.append("\n".join(parts))
    return "\n".join(blocks)


def documents_to_xml(results: list[MultiGetFile]) -> str:
    lines = ['<?xml version="1.0" encoding="UTF-8"?>', "<documents>"]
    for r in results:
        docid_attr = f' docid="#{r.docid}"' if r.docid else ""
        lines.append(f"  <document{docid_attr}>")
        lines.append(f"    <file>{escape_xml(r.display_path)}</file>")
        lines.append(f"    <title>{escape_xml(r.title)}</title>")
        if r.context:
            lines.append(f"    <context>{escape_xml(r.context)}</context>")
        if r.skipped:
            lines.append("    <skipped>true</skipped>")
            lines.append(f"    <reason>{escape_xml(r.skip_reason or '')}</reason>")
        else:
            lines.append(f"    <body>{escape_xml(r.body)}</body>")
        lines.append("  </document>")
    lines.append("</documents>")
    return "\n".join(lines)


def documents_to_toon(results: list[MultiGetFile]) -> str:
    field_names = ["docid", "file", "title", "context", "skipped", "body"]
    rows: list[list[object]] = [
        [
            f"#{r.docid}" if r.docid else "",
            r.display_path,
            r.title,
            r.context or "",
            r.skipped,
            r.skip_reason if r.skipped else r.body,
        ]
        for r in results
    ]
    return _to_toon_table("documents", field_names, rows)


def format_documents(results: list[MultiGetFile], out_format: OutputFormat) -> str:
    if out_format == "json":
        return documents_to_json(results)
    if out_format == "csv":
        return documents_to_csv(results)
    if out_format == "files":
        return documents_to_files(results)
    if out_format == "xml":
        return documents_to_xml(results)
    if out_format == "toon":
        return documents_to_toon(results)
    # "cli" falls back to markdown, same as the TS reference.
    return documents_to_markdown(results)


# =============================================================================
# Single document (`get`'s own header+body rendering is plain text, not
# routed through any of these - `get` has no --format flag in the TS
# reference. These exist for reuse by the MCP resource handler, Phase 9).
# =============================================================================


def document_to_json(doc: DocumentDetail) -> str:
    entry: dict[str, object] = {"file": doc.display_path, "title": doc.title}
    if doc.context:
        entry["context"] = doc.context
    entry["hash"] = doc.hash
    entry["modifiedAt"] = doc.modified_at.isoformat()
    entry["bodyLength"] = doc.body_length
    entry["body"] = doc.body
    return json.dumps(entry, indent=2)


def document_to_markdown(doc: DocumentDetail) -> str:
    parts = [f"# {doc.title or doc.display_path}\n"]
    if doc.context:
        parts.append(f"**Context:** {doc.context}\n")
    parts.append(f"**File:** {doc.display_path}\n")
    parts.append(f"**Modified:** {doc.modified_at.isoformat()}\n")
    parts.append(f"---\n\n{doc.body}\n")
    return "\n".join(parts)


def document_to_xml(doc: DocumentDetail) -> str:
    lines = ['<?xml version="1.0" encoding="UTF-8"?>', "<document>"]
    lines.append(f"  <file>{escape_xml(doc.display_path)}</file>")
    lines.append(f"  <title>{escape_xml(doc.title)}</title>")
    if doc.context:
        lines.append(f"  <context>{escape_xml(doc.context)}</context>")
    lines.append(f"  <hash>{escape_xml(doc.hash)}</hash>")
    lines.append(f"  <modifiedAt>{escape_xml(doc.modified_at.isoformat())}</modifiedAt>")
    lines.append(f"  <bodyLength>{doc.body_length}</bodyLength>")
    lines.append(f"  <body>{escape_xml(doc.body)}</body>")
    lines.append("</document>")
    return "\n".join(lines)


def format_document(doc: DocumentDetail, out_format: OutputFormat) -> str:
    if out_format == "json":
        return document_to_json(doc)
    if out_format == "xml":
        return document_to_xml(doc)
    return document_to_markdown(doc)
