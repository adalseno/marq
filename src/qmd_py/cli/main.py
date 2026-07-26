"""qmdpy CLI entry point.

Command groups are added phase by phase (see the project plan) - Phase 9
adds the MCP server, Phase 10 adds bench/doctor/skill.
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
from qmd_py.cli.commands.write import (
    cleanup_command,
    collection_group,
    context_group,
    embed_command,
    update_command,
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
cli.add_command(collection_group)
cli.add_command(update_command)
cli.add_command(embed_command)
cli.add_command(cleanup_command)
cli.add_command(context_group)


if __name__ == "__main__":
    cli()
