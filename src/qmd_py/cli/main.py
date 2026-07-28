"""marq CLI entry point.

Command groups are added phase by phase (see the project plan) - Phase 9
adds the MCP server, Phase 10 adds bench/doctor/skill.
"""

import sys

import click
from pydantic import ValidationError

from qmd_py.cli.commands.bench import bench_command
from qmd_py.cli.commands.doctor import doctor_command
from qmd_py.cli.commands.mcp import mcp_group
from qmd_py.cli.commands.query import query_command
from qmd_py.cli.commands.read import (
    get_command,
    ls_command,
    multi_get_command,
    search_command,
    status_command,
    vsearch_command,
)
from qmd_py.cli.commands.skill import skill_command, skills_group
from qmd_py.cli.commands.write import (
    cleanup_command,
    collection_group,
    context_group,
    embed_command,
    update_command,
)
from qmd_py.config import get_settings
from qmd_py.log import setup_logging


@click.group()
@click.version_option()
@click.option(
    "-v",
    "--verbose",
    count=True,
    help="Log to stderr: -v for INFO (per-query timings, per-collection counts), -vv for DEBUG.",
)
def cli(verbose: int) -> None:
    """marq: centralized markdown/code search over Postgres/pgvector."""
    # Settings are read lazily (and may be invalid - main() reports that
    # below), so a -v/-vv flag must be able to configure logging without
    # them. Without a flag, logging is configured from settings at the
    # first command that touches them, or left at the WARNING default.
    if verbose:
        setup_logging("DEBUG" if verbose > 1 else "INFO", force=True)
    else:
        try:
            settings = get_settings()
        except ValidationError:
            return
        setup_logging(settings.log_level, settings.log_file)


cli.add_command(search_command)
cli.add_command(vsearch_command)
cli.add_command(query_command)
cli.add_command(query_command, name="deep-search")
cli.add_command(get_command)
cli.add_command(multi_get_command)
cli.add_command(ls_command)
cli.add_command(status_command)
cli.add_command(collection_group)
cli.add_command(update_command)
cli.add_command(embed_command)
cli.add_command(cleanup_command)
cli.add_command(context_group)
cli.add_command(mcp_group)
cli.add_command(doctor_command)
cli.add_command(bench_command)
cli.add_command(skill_command)
cli.add_command(skills_group)


def main() -> None:
    """Console-script entry point (see pyproject.toml) - wraps `cli()` so
    a missing/invalid MARQ_* setting turns into a short, actionable
    message instead of a raw pydantic traceback. `get_settings()` is
    called lazily from deep inside individual commands' (often async)
    bodies, not once up front, so this outermost wrapper is the one
    place that reliably sees the error regardless of which command
    triggered it."""
    try:
        cli()
    except ValidationError as exc:
        click.echo("marq: invalid or missing configuration", err=True)
        for error in exc.errors():
            env_var = f"MARQ_{'.'.join(str(p) for p in error['loc']).upper()}"
            click.echo(f"  {env_var}: {error['msg']}", err=True)
        click.echo(
            "\nSet these as environment variables or in a .env file in the "
            "current directory.",
            err=True,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
