"""qmdpy CLI entry point.

Command groups are added phase by phase (see the project plan) — this is
intentionally a bare skeleton until Phase 6/7 wire up the real commands.
"""

import click


@click.group()
@click.version_option()
def cli() -> None:
    """qmd-py: centralized markdown search over Postgres/pgvector."""


if __name__ == "__main__":
    cli()
