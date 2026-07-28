"""Write/indexing CLI commands: collection management, update, embed,
cleanup, context - ports of the TS reference's `src/cli/qmd.ts` command
handlers, adapted to qmd-py's real Collection ownership (no YAML config).
"""

import subprocess
from pathlib import Path

import click
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from qmd_py.auth import CurrentUser
from qmd_py.cli.runtime import run
from qmd_py.config import get_settings
from qmd_py.llm.client import LlmClient
from qmd_py.search.vector import (
    cleanup_orphaned_embeddings,
    embed_pending_documents,
    get_or_probe_dimension,
)
from qmd_py.store import (
    CollectionNotFoundError,
    add_context,
    cleanup_orphaned_content,
    context_check,
    delete_inactive_documents,
    delete_llm_cache,
    get_collection,
    get_global_context,
    list_collections,
    list_contexts,
    reindex_collection,
    remove_collection,
    remove_context,
    rename_collection,
    set_global_context,
    set_include_by_default,
    set_update_command,
)
from qmd_py.store import add_collection as store_add_collection
from qmd_py.vpath import parse_virtual_path


def _parse_context_path(path: str) -> tuple[str, str]:
    if path.startswith("marq://"):
        parsed = parse_virtual_path(path)
        if parsed:
            return parsed
    parts = path.strip("/").split("/", 1)
    return parts[0], (parts[1] if len(parts) > 1 else "")


# =============================================================================
# collection
# =============================================================================


@click.group("collection")
def collection_group() -> None:
    """Manage indexed collections."""


@collection_group.command("add")
@click.argument("path")
@click.option("--name", default=None, help="Collection name (default: directory basename)")
@click.option("--mask", "pattern", default="**/*.md", show_default=True, help="Glob pattern")
def collection_add_command(path: str, name: str | None, pattern: str) -> None:
    """Add and index a collection."""
    resolved_path = str(Path(path).expanduser().resolve())
    collection_name = name or Path(resolved_path).name or "root"
    run(_collection_add_impl, resolved_path, pattern, collection_name)


async def _collection_add_impl(
    session: AsyncSession, user: CurrentUser, path: str, pattern: str, name: str
) -> None:
    click.echo(f"Creating collection '{name}'...")
    try:
        await store_add_collection(session, user, name, path, pattern)
        await session.commit()
    except IntegrityError:
        await session.rollback()
        click.echo(f"Collection '{name}' already exists.", err=True)
        click.echo("Use a different name with --name <name>", err=True)
        raise SystemExit(1) from None

    result = await reindex_collection(session, user, name)
    await session.commit()
    click.echo(f"✓ Collection '{name}' created successfully")
    click.echo(
        f"  Indexed: {result.indexed} new, {result.updated} updated, "
        f"{result.unchanged} unchanged, {result.removed} removed"
    )
    if result.skipped_oversize:
        click.echo(
            f"  Skipped {result.skipped_oversize} oversized file(s) "
            "(over ~1 MB - too large to index)"
        )


@collection_group.command("remove")
@click.argument("name")
def collection_remove_command(name: str) -> None:
    """Remove a collection by name."""
    run(_collection_remove_impl, name)


async def _collection_remove_impl(session: AsyncSession, user: CurrentUser, name: str) -> None:
    try:
        result = await remove_collection(session, user, name)
        await session.commit()
    except CollectionNotFoundError:
        click.echo(f"Collection not found: {name}", err=True)
        click.echo("Run 'marq collection list' to see available collections.", err=True)
        raise SystemExit(1) from None
    click.echo(f"✓ Removed collection '{name}'")
    click.echo(f"  Deleted {result.deleted_docs} documents")
    if result.cleaned_hashes:
        click.echo(f"  Cleaned up {result.cleaned_hashes} orphaned content hashes")


@collection_group.command("rename")
@click.argument("old_name")
@click.argument("new_name")
def collection_rename_command(old_name: str, new_name: str) -> None:
    """Rename a collection."""
    run(_collection_rename_impl, old_name, new_name)


async def _collection_rename_impl(
    session: AsyncSession, user: CurrentUser, old_name: str, new_name: str
) -> None:
    try:
        await rename_collection(session, user, old_name, new_name)
        await session.commit()
    except CollectionNotFoundError:
        click.echo(f"Collection not found: {old_name}", err=True)
        raise SystemExit(1) from None
    click.echo(f"✓ Renamed collection '{old_name}' to '{new_name}'")


@collection_group.command("list")
def collection_list_command() -> None:
    """List all collections with details."""
    run(_collection_list_impl)


async def _collection_list_impl(session: AsyncSession, user: CurrentUser) -> None:
    rows = await list_collections(session, user)
    if not rows:
        click.echo("No collections found. Run 'marq collection add .' to create one.")
        return
    click.echo(f"Collections ({len(rows)}):\n")
    for c in rows:
        excluded = "" if c.include_by_default else " [excluded]"
        click.echo(f"{c.name} (marq://{c.name}/){excluded}")
        click.echo(f"  Pattern:  {c.pattern}")
        click.echo(f"  Files:    {c.active_count}")
        updated = c.last_modified.strftime("%Y-%m-%d") if c.last_modified else "never"
        click.echo(f"  Updated:  {updated}\n")


@collection_group.command("show")
@click.argument("name")
def collection_show_command(name: str) -> None:
    """Show collection details."""
    run(_collection_show_impl, name)


async def _collection_show_impl(session: AsyncSession, user: CurrentUser, name: str) -> None:
    try:
        collection = await get_collection(session, user, name)
    except CollectionNotFoundError:
        click.echo(f"Collection not found: {name}", err=True)
        raise SystemExit(1) from None
    click.echo(f"Collection: {collection.name}")
    click.echo(f"  Path:     {collection.path}")
    click.echo(f"  Pattern:  {collection.pattern}")
    click.echo(f"  Include:  {'yes (default)' if collection.include_by_default else 'no'}")
    if collection.update_command:
        click.echo(f"  Update:   {collection.update_command}")
    contexts = await list_contexts(session, user)
    context_count = sum(1 for c in contexts if c.collection == name)
    if context_count:
        click.echo(f"  Contexts: {context_count}")


@collection_group.command("update-cmd")
@click.argument("name")
@click.argument("command_parts", nargs=-1)
def collection_update_cmd_command(name: str, command_parts: tuple[str, ...]) -> None:
    """Set (or clear, if omitted) the pre-update command for a collection."""
    command = " ".join(command_parts) if command_parts else None
    run(_collection_update_cmd_impl, name, command)


async def _collection_update_cmd_impl(
    session: AsyncSession, user: CurrentUser, name: str, command: str | None
) -> None:
    try:
        await set_update_command(session, user, name, command)
        await session.commit()
    except CollectionNotFoundError:
        click.echo(f"Collection not found: {name}", err=True)
        raise SystemExit(1) from None
    if command:
        click.echo(f"✓ Set update command for '{name}': {command}")
    else:
        click.echo(f"✓ Cleared update command for '{name}'")


@collection_group.command("include")
@click.argument("name")
def collection_include_command(name: str) -> None:
    """Include a collection in default (unscoped) queries."""
    run(_collection_set_default_impl, name, include=True)


@collection_group.command("exclude")
@click.argument("name")
def collection_exclude_command(name: str) -> None:
    """Exclude a collection from default (unscoped) queries."""
    run(_collection_set_default_impl, name, include=False)


async def _collection_set_default_impl(
    session: AsyncSession, user: CurrentUser, name: str, *, include: bool
) -> None:
    try:
        await set_include_by_default(session, user, name, include)
        await session.commit()
    except CollectionNotFoundError:
        click.echo(f"Collection not found: {name}", err=True)
        raise SystemExit(1) from None
    verb = "included in" if include else "excluded from"
    click.echo(f"✓ Collection '{name}' {verb} default queries")


# =============================================================================
# update
# =============================================================================


@click.command("update")
def update_command() -> None:
    """Re-index all collections; configured update commands run first."""
    run(_update_impl)


async def _update_impl(session: AsyncSession, user: CurrentUser) -> None:
    """One collection's failure doesn't abort the rest.

    This command invites cron-style usage, where aborting on the first
    failing update command would silently leave every later collection
    unindexed. Failures are collected and reported together at the end,
    and the exit code is still non-zero so a wrapper can react. A
    collection whose update command failed is not reindexed - its source
    tree may be half-updated - but the following ones still are.
    """
    rows = await list_collections(session, user)
    if not rows:
        click.echo("No collections found. Run 'marq collection add .' to index files.")
        return

    failures: list[tuple[str, str]] = []
    click.echo(f"Updating {len(rows)} collection(s)...\n")
    for i, row in enumerate(rows, start=1):
        click.echo(f"[{i}/{len(rows)}] {row.name} ({row.pattern})")
        collection = await get_collection(session, user, row.name)

        if collection.update_command:
            click.echo(f"    Running update command: {collection.update_command}")
            proc = subprocess.run(  # noqa: S602 - user's own configured command, same trust
                collection.update_command,
                shell=True,
                cwd=collection.path,
                capture_output=True,
                text=True,
                check=False,
            )
            for line in proc.stdout.splitlines():
                click.echo(f"    {line}")
            for line in proc.stderr.splitlines():
                click.echo(f"    {line}")
            if proc.returncode != 0:
                click.echo(f"✗ Update command failed with exit code {proc.returncode}", err=True)
                failures.append((row.name, f"update command exited {proc.returncode}"))
                click.echo()
                continue

        try:
            result = await reindex_collection(session, user, row.name)
            await session.commit()
        except (OSError, SQLAlchemyError) as exc:
            # OSError: realistically a vanished/unreadable source
            # directory. SQLAlchemyError: a per-file database surprise
            # (the oversized-tsvector class of failure) - either way, one
            # broken collection shouldn't strand the others.
            await session.rollback()
            click.echo(f"✗ Reindex failed: {exc}", err=True)
            failures.append((row.name, f"reindex failed: {exc}"))
            click.echo()
            continue

        click.echo(
            f"Indexed: {result.indexed} new, {result.updated} updated, "
            f"{result.unchanged} unchanged, {result.removed} removed"
        )
        if result.skipped_oversize:
            click.echo(
                f"Skipped {result.skipped_oversize} oversized file(s) "
                "(over ~1 MB - too large to index)"
            )
        if result.orphaned_cleaned:
            click.echo(f"Cleaned up {result.orphaned_cleaned} orphaned content hash(es)")
        click.echo()

    if failures:
        click.echo(f"✗ {len(failures)} of {len(rows)} collection(s) failed:", err=True)
        for name, reason in failures:
            click.echo(f"    {name}: {reason}", err=True)
        raise SystemExit(1)

    click.echo("✓ All collections updated.")


# =============================================================================
# embed
# =============================================================================


@click.command("embed")
@click.option("-c", "--collection", default=None, help="Restrict embedding to one collection")
@click.option("--model", default=None, help="Embedding model slug (default: the configured model)")
def embed_command(collection: str | None, model: str | None) -> None:
    """Generate vector embeddings for pending documents."""
    run(_embed_impl, collection=collection, model=model)


async def _embed_impl(
    session: AsyncSession, user: CurrentUser, *, collection: str | None, model: str | None
) -> None:
    if collection is not None:
        try:
            await get_collection(session, user, collection)
        except CollectionNotFoundError:
            click.echo(f"Collection not found: {collection}", err=True)
            raise SystemExit(1) from None

    settings = get_settings()
    model_slug = model or settings.embed_model
    async with LlmClient(settings.llm_base_url) as llm_client:
        dimension = await get_or_probe_dimension(session, llm_client, model_slug)
        # Commits per document internally, so an interrupted run keeps
        # what it already embedded and resumes where it left off.
        result = await embed_pending_documents(
            session, user, llm_client, model_slug, dimension, collection_name=collection
        )
    await session.commit()
    click.echo(f"✓ Embedded {result.docs_processed} document(s), {result.chunks_embedded} chunk(s)")


# =============================================================================
# cleanup
# =============================================================================


@click.command("cleanup")
def cleanup_command() -> None:
    """Remove orphaned content/embeddings and inactive document records.

    No VACUUM step (unlike the TS reference): Postgres's autovacuum
    handles space reclaim automatically, and a manual VACUUM on a shared
    server is more likely to cause lock contention than help.
    """
    run(_cleanup_impl)


async def _cleanup_impl(session: AsyncSession, user: CurrentUser) -> None:
    cache_count = await delete_llm_cache(session)
    click.echo(f"✓ Cleared {cache_count} cached API responses")

    orphaned_vectors = await cleanup_orphaned_embeddings(session)
    if orphaned_vectors:
        click.echo(f"✓ Removed {orphaned_vectors} orphaned embedding chunks")
    else:
        click.echo("No orphaned embeddings to remove")

    inactive = await delete_inactive_documents(session, user)
    if inactive:
        click.echo(f"✓ Removed {inactive} inactive document records")

    orphaned_content = await cleanup_orphaned_content(session)
    if orphaned_content:
        click.echo(f"✓ Removed {orphaned_content} orphaned content hash(es)")

    await session.commit()


# =============================================================================
# context
# =============================================================================


@click.group("context")
def context_group() -> None:
    """Manage per-path and global search context."""


@context_group.command("add")
@click.argument("path")
@click.argument("text")
def context_add_command(path: str, text: str) -> None:
    """Add context for a path (or "/" for global context)."""
    run(_context_add_impl, path, text)


async def _context_add_impl(session: AsyncSession, user: CurrentUser, path: str, text: str) -> None:
    if path == "/":
        await set_global_context(session, user, text)
        await session.commit()
        click.echo("✓ Set global context")
        return
    collection_name, path_prefix = _parse_context_path(path)
    try:
        await add_context(session, user, collection_name, path_prefix, text)
        await session.commit()
    except CollectionNotFoundError:
        click.echo(f"Collection not found: {collection_name}", err=True)
        raise SystemExit(1) from None
    click.echo(f"✓ Set context for marq://{collection_name}/{path_prefix}")


@context_group.command("list")
def context_list_command() -> None:
    """List all contexts."""
    run(_context_list_impl)


async def _context_list_impl(session: AsyncSession, user: CurrentUser) -> None:
    global_ctx = await get_global_context(session, user)
    rows = await list_contexts(session, user)
    if not rows and not global_ctx:
        click.echo("No contexts found.")
        return
    if global_ctx:
        click.echo(f"Global: {global_ctx}\n")
    for r in rows:
        label = f"marq://{r.collection}/{r.path}" if r.path else f"marq://{r.collection}/"
        click.echo(f"{label}\n  {r.context}\n")


@context_group.command("rm")
@click.argument("path")
def context_rm_command(path: str) -> None:
    """Remove context for a path (or "/" for global context)."""
    run(_context_rm_impl, path)


async def _context_rm_impl(session: AsyncSession, user: CurrentUser, path: str) -> None:
    if path == "/":
        await set_global_context(session, user, None)
        await session.commit()
        click.echo("✓ Removed global context")
        return
    collection_name, path_prefix = _parse_context_path(path)
    try:
        removed = await remove_context(session, user, collection_name, path_prefix)
        await session.commit()
    except CollectionNotFoundError:
        click.echo(f"Collection not found: {collection_name}", err=True)
        raise SystemExit(1) from None
    if removed:
        click.echo(f"✓ Removed context for marq://{collection_name}/{path_prefix}")
    else:
        click.echo(f"No context found for marq://{collection_name}/{path_prefix}", err=True)
        raise SystemExit(1)


@context_group.command("check")
def context_check_command() -> None:
    """Check for collections/paths missing context."""
    run(_context_check_impl)


async def _context_check_impl(session: AsyncSession, user: CurrentUser) -> None:
    missing_collections, missing_paths = await context_check(session, user)
    if not missing_collections and not missing_paths:
        click.echo("✓ All collections have context configured.")
        return
    if missing_collections:
        click.echo("Collections without any context:")
        for c in missing_collections:
            click.echo(f"  {c.name} ({c.doc_count} docs)")
        click.echo()
    if missing_paths:
        click.echo("Collections with uncovered top-level paths:")
        for name, paths in missing_paths.items():
            click.echo(f"  {name}: {', '.join(paths)}")
