"""Read-only CLI commands: search, vsearch, get, multi-get, ls, status -
ports of the TS reference's `src/cli/qmd.ts` command handlers, adapted to
qmd-py's real Collection ownership (no YAML config) and single-user ACL
mock (see auth.py).
"""

import re

import click
from sqlalchemy.ext.asyncio import AsyncSession

from qmd_py.auth import CurrentUser
from qmd_py.cli.formatter import FormatOptions, format_documents, format_search_results
from qmd_py.cli.runtime import run
from qmd_py.config import get_settings
from qmd_py.llm.client import LlmClient
from qmd_py.search.fts import SearchResult, search_fts
from qmd_py.search.vector import search_vec
from qmd_py.store import (
    DEFAULT_MULTI_GET_MAX_BYTES,
    CollectionNotFoundError,
    DocumentNotFound,
    add_line_numbers,
    find_document,
    get_status,
    list_collections,
    list_files,
    multi_get,
)
from qmd_py.vpath import parse_virtual_path

_FORMAT_CHOICES = ("cli", "json", "csv", "md", "xml", "files", "toon")


def _format_options(fn: click.decorators.FC) -> click.decorators.FC:
    """Shared --format plus legacy --json/--csv/--md/--xml/--files aliases
    (still work as aliases, per the frozen CLI surface)."""
    fn = click.option(
        "--format",
        "out_format",
        type=click.Choice(_FORMAT_CHOICES),
        default="cli",
        show_default=True,
    )(fn)
    fn = click.option("--json", "json_flag", is_flag=True, hidden=True)(fn)
    fn = click.option("--csv", "csv_flag", is_flag=True, hidden=True)(fn)
    fn = click.option("--md", "md_flag", is_flag=True, hidden=True)(fn)
    fn = click.option("--xml", "xml_flag", is_flag=True, hidden=True)(fn)
    fn = click.option("--files", "files_flag", is_flag=True, hidden=True)(fn)
    return fn


def _resolve_format(
    out_format: str,
    json_flag: bool,
    csv_flag: bool,
    md_flag: bool,
    xml_flag: bool,
    files_flag: bool,
) -> str:
    if json_flag:
        return "json"
    if csv_flag:
        return "csv"
    if md_flag:
        return "md"
    if xml_flag:
        return "xml"
    if files_flag:
        return "files"
    return out_format


async def _resolve_collection_names(
    session: AsyncSession, user: CurrentUser, collections: tuple[str, ...]
) -> list[str]:
    """Explicit `-c` names win outright; with none given, falls back to
    collections marked `include_by_default` - port of the TS reference's
    `resolveCollectionFilter`."""
    if collections:
        return list(collections)
    rows = await list_collections(session, user)
    return [c.name for c in rows if c.include_by_default]


def _echo_empty_search_results(out_format: str) -> None:
    if out_format == "cli":
        click.echo("No results found.")
    else:
        click.echo(format_search_results([], out_format))


def _filter_by_collections(results: list[SearchResult], names: list[str]) -> list[SearchResult]:
    """Post-filter mirroring the TS reference's `filterByCollections`: a
    single (or zero) collection name is a no-op here since the DB query
    already scoped to it; multiple names filter the combined result set."""
    if len(names) <= 1:
        return results
    prefixes = tuple(f"qmd://{n}/" for n in names)
    return [r for r in results if r.filepath.startswith(prefixes)]


# =============================================================================
# search
# =============================================================================


@click.command("search")
@click.argument("query")
@click.option("-c", "--collection", "collections", multiple=True, help="Restrict to collection(s)")
@click.option("-n", "limit", type=int, default=10, show_default=True, help="Number of results")
@click.option("--all", "show_all", is_flag=True, help="Return all matches")
@click.option("--min-score", type=float, default=0.0, help="Minimum score threshold")
@click.option("--full", is_flag=True, help="Show full document content")
@click.option("--intent", default=None, help="Describe what you're after to sharpen ranking")
@_format_options
def search_command(
    query: str,
    collections: tuple[str, ...],
    limit: int,
    show_all: bool,
    min_score: float,
    full: bool,
    intent: str | None,
    out_format: str,
    json_flag: bool,
    csv_flag: bool,
    md_flag: bool,
    xml_flag: bool,
    files_flag: bool,
) -> None:
    """Full-text keyword search (BM25, no LLM)."""
    out_format = _resolve_format(out_format, json_flag, csv_flag, md_flag, xml_flag, files_flag)
    run(
        _search_impl,
        query,
        collections=collections,
        limit=limit,
        show_all=show_all,
        min_score=min_score,
        full=full,
        intent=intent,
        out_format=out_format,
    )


async def _search_impl(
    session: AsyncSession,
    user: CurrentUser,
    query: str,
    *,
    collections: tuple[str, ...],
    limit: int,
    show_all: bool,
    min_score: float,
    full: bool,
    intent: str | None,
    out_format: str,
) -> None:
    names = await _resolve_collection_names(session, user, collections)
    single = names[0] if len(names) == 1 else None
    fetch_limit = 100_000 if show_all else max(50, limit * 2)

    results = await search_fts(session, user, query, limit=fetch_limit, collection_name=single)
    results = _filter_by_collections(results, names)
    results = [r for r in results if r.score >= min_score]
    if not show_all:
        results = results[:limit]

    if not results:
        _echo_empty_search_results(out_format)
        return
    opts = FormatOptions(full=full, query=query, line_numbers=False, intent=intent)
    click.echo(format_search_results(results, out_format, opts))


# =============================================================================
# vsearch
# =============================================================================


@click.command("vsearch")
@click.argument("query")
@click.option("-c", "--collection", "collections", multiple=True, help="Restrict to collection(s)")
@click.option("-n", "limit", type=int, default=10, show_default=True, help="Number of results")
@click.option("--all", "show_all", is_flag=True, help="Return all matches")
@click.option("--min-score", type=float, default=None, help="Minimum score threshold (default 0.3)")
@click.option("--full", is_flag=True, help="Show full document content")
@click.option("--intent", default=None, help="Describe what you're after to sharpen ranking")
@_format_options
def vsearch_command(
    query: str,
    collections: tuple[str, ...],
    limit: int,
    show_all: bool,
    min_score: float | None,
    full: bool,
    intent: str | None,
    out_format: str,
    json_flag: bool,
    csv_flag: bool,
    md_flag: bool,
    xml_flag: bool,
    files_flag: bool,
) -> None:
    """Vector similarity search (no reranking)."""
    out_format = _resolve_format(out_format, json_flag, csv_flag, md_flag, xml_flag, files_flag)
    resolved_min_score = min_score if min_score is not None else 0.3
    run(
        _vsearch_impl,
        query,
        collections=collections,
        limit=limit,
        show_all=show_all,
        min_score=resolved_min_score,
        full=full,
        intent=intent,
        out_format=out_format,
    )


async def _vsearch_impl(
    session: AsyncSession,
    user: CurrentUser,
    query: str,
    *,
    collections: tuple[str, ...],
    limit: int,
    show_all: bool,
    min_score: float,
    full: bool,
    intent: str | None,
    out_format: str,
) -> None:
    names = await _resolve_collection_names(session, user, collections)
    single = names[0] if len(names) == 1 else None
    fetch_limit = 100_000 if show_all else max(50, limit * 2)
    settings = get_settings()

    async with LlmClient(settings.llm_base_url) as llm_client:
        results = await search_vec(
            session, user, query, llm_client, settings.embed_model,
            limit=fetch_limit, collection_name=single,
        )
    results = _filter_by_collections(results, names)
    results = [r for r in results if r.score >= min_score]
    if not show_all:
        results = results[:limit]

    if not results:
        _echo_empty_search_results(out_format)
        return
    opts = FormatOptions(full=full, query=query, line_numbers=False, intent=intent)
    click.echo(format_search_results(results, out_format, opts))


# =============================================================================
# get
# =============================================================================

_PATH_RANGE_RE = re.compile(r":(\d+):(\d+)$")
_PATH_LINE_RE = re.compile(r":(\d+)$")


@click.command("get")
@click.argument("filepath")
@click.option("--from", "from_line", type=int, default=None, help="Starting line number")
@click.option("-l", "max_lines", type=int, default=None, help="Maximum lines to show")
@click.option("--no-line-numbers", is_flag=True, help="Disable line numbers")
def get_command(
    filepath: str, from_line: int | None, max_lines: int | None, no_line_numbers: bool
) -> None:
    """Get a document by path or docid; optional line range (path:from:count)."""
    run(
        _get_impl,
        filepath,
        from_line=from_line,
        max_lines=max_lines,
        line_numbers=not no_line_numbers,
    )


async def _get_impl(
    session: AsyncSession,
    user: CurrentUser,
    filepath: str,
    *,
    from_line: int | None,
    max_lines: int | None,
    line_numbers: bool,
) -> None:
    name = filepath
    match = _PATH_RANGE_RE.search(name)
    if match:
        if from_line is None:
            from_line = int(match.group(1))
        if max_lines is None:
            max_lines = int(match.group(2))
        name = name[: match.start()]
    else:
        match = _PATH_LINE_RE.search(name)
        if match:
            if from_line is None:
                from_line = int(match.group(1))
            name = name[: match.start()]

    if from_line is not None:
        from_line = max(1, from_line)

    doc = await find_document(session, user, name)
    if isinstance(doc, DocumentNotFound):
        click.echo(f"Document not found: {doc.query}", err=True)
        if doc.similar_files:
            click.echo("Similar files:", err=True)
            for similar in doc.similar_files:
                click.echo(f"  {similar}", err=True)
        raise SystemExit(1)

    canonical_path = f"qmd://{doc.display_path}"
    header = f"{canonical_path}  #{doc.docid}"

    output = doc.body
    start_line = from_line or 1
    if from_line is not None or max_lines is not None:
        lines = output.split("\n")
        start = start_line - 1
        end = start + max_lines if max_lines is not None else len(lines)
        output = "\n".join(lines[start:end])

    if line_numbers:
        output = add_line_numbers(output, start_line)

    click.echo(header)
    if doc.context:
        click.echo(f"Folder Context: {doc.context}")
    click.echo("---\n")
    click.echo(output)


# =============================================================================
# multi-get
# =============================================================================


@click.command("multi-get")
@click.argument("pattern")
@click.option("-l", "max_lines", type=int, default=None, help="Maximum lines per file")
@click.option(
    "--max-bytes", type=int, default=DEFAULT_MULTI_GET_MAX_BYTES, show_default=True,
    help="Skip files larger than this",
)
@click.option("--no-line-numbers", is_flag=True, help="Disable line numbers")
@_format_options
def multi_get_command(
    pattern: str,
    max_lines: int | None,
    max_bytes: int,
    no_line_numbers: bool,
    out_format: str,
    json_flag: bool,
    csv_flag: bool,
    md_flag: bool,
    xml_flag: bool,
    files_flag: bool,
) -> None:
    """Get multiple documents by glob pattern or comma-separated list."""
    out_format = _resolve_format(out_format, json_flag, csv_flag, md_flag, xml_flag, files_flag)
    run(
        _multi_get_impl,
        pattern,
        max_lines=max_lines,
        max_bytes=max_bytes,
        line_numbers=not no_line_numbers,
        out_format=out_format,
    )


async def _multi_get_impl(
    session: AsyncSession,
    user: CurrentUser,
    pattern: str,
    *,
    max_lines: int | None,
    max_bytes: int,
    line_numbers: bool,
    out_format: str,
) -> None:
    results = await multi_get(
        session, user, pattern, max_lines=max_lines, max_bytes=max_bytes, line_numbers=line_numbers
    )
    if not results:
        click.echo(f"No files matched pattern: {pattern}", err=True)
        raise SystemExit(1)
    click.echo(format_documents(results, out_format))


# =============================================================================
# ls
# =============================================================================


@click.command("ls")
@click.argument("path_arg", required=False)
def ls_command(path_arg: str | None) -> None:
    """List collections, or files in a collection[/path]."""
    run(_ls_impl, path_arg)


async def _ls_impl(session: AsyncSession, user: CurrentUser, path_arg: str | None) -> None:
    if not path_arg:
        collections = await list_collections(session, user)
        if not collections:
            click.echo("No collections found. Run 'qmdpy collection add .' to index files.")
            return
        click.echo("Collections:")
        for c in collections:
            click.echo(f"  qmd://{c.name}/  ({c.active_count} files)")
        return

    if path_arg.startswith("qmd://"):
        parsed = parse_virtual_path(path_arg)
        collection_name, path_prefix = parsed if parsed else (path_arg, "")
    else:
        parts = path_arg.split("/", 1)
        collection_name, path_prefix = parts[0], (parts[1] if len(parts) > 1 else "")

    try:
        files = await list_files(session, user, collection_name, path_prefix or None)
    except CollectionNotFoundError:
        click.echo(f"Collection not found: {collection_name}", err=True)
        click.echo("Run 'qmdpy ls' to see available collections.", err=True)
        raise SystemExit(1) from None

    if not files:
        if path_prefix:
            click.echo(f"No files found under qmd://{collection_name}/{path_prefix}")
        else:
            click.echo(f"No files found in collection: {collection_name}")
        return

    for f in files:
        size_str = f"{f.size}B" if f.size < 1024 else f"{f.size / 1024:.1f}KB"
        when = f.modified_at.strftime("%Y-%m-%d %H:%M")
        click.echo(f"  {size_str:>8}  {when}  qmd://{collection_name}/{f.path}")


# =============================================================================
# status
# =============================================================================


@click.command("status")
def status_command() -> None:
    """Show index status and collections."""
    run(_status_impl)


async def _status_impl(session: AsyncSession, user: CurrentUser) -> None:
    status = await get_status(session, user)
    click.echo("QMD Status\n")
    click.echo(f"Documents\n  Total: {status.total_documents} files indexed\n")
    if not status.collections:
        click.echo("No collections found. Run 'qmdpy collection add .' to index files.")
        return
    click.echo("Collections:")
    for c in status.collections:
        updated = c.last_updated.strftime("%Y-%m-%d") if c.last_updated else "never"
        click.echo(f"  {c.name} (qmd://{c.name}/)")
        click.echo(f"    Pattern:  {c.pattern}")
        click.echo(f"    Files:    {c.doc_count} (updated {updated})")
