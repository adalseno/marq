"""qmdpy CLI entry point.

Command groups are added phase by phase (see the project plan) - Phase 7
adds the write/indexing commands (collection management, update, embed,
context) that fill out the rest of the frozen CLI surface.
"""

import click

from qmd_py.cli.commands.read import (
    get_command,
    ls_command,
    multi_get_command,
    search_command,
    status_command,
    vsearch_command,
)


@click.group()
@click.version_option()
def cli() -> None:
    """qmd-py: centralized markdown search over Postgres/pgvector."""


cli.add_command(search_command)
cli.add_command(vsearch_command)
cli.add_command(get_command)
cli.add_command(multi_get_command)
cli.add_command(ls_command)
cli.add_command(status_command)


if __name__ == "__main__":
    cli()
