"""`bench` - run search-quality benchmarks against a fixture file - port
of the TS reference's `bench` CLI command (src/cli/qmd.ts).
"""

import click
from sqlalchemy.ext.asyncio import AsyncSession

from qmd_py.auth import CurrentUser
from qmd_py.bench import (
    bench_result_to_json,
    format_bench_summary,
    format_bench_table,
    run_benchmark,
)
from qmd_py.cli.runtime import run
from qmd_py.config import get_settings
from qmd_py.llm.client import LlmClient


@click.command("bench")
@click.argument("fixture_path", type=click.Path(exists=True, dir_okay=False))
@click.option("-c", "--collection", default=None, help="Restrict to one collection")
@click.option(
    "--format", "out_format", type=click.Choice(["cli", "json"]), default="cli", show_default=True
)
@click.option("--json", "json_flag", is_flag=True, hidden=True)
def bench_command(
    fixture_path: str, collection: str | None, out_format: str, json_flag: bool
) -> None:
    """Run search-quality benchmarks (precision/recall/MRR/F1) against a fixture file."""
    if json_flag:
        out_format = "json"
    run(_bench_impl, fixture_path, collection=collection, out_format=out_format)


async def _bench_impl(
    session: AsyncSession,
    user: CurrentUser,
    fixture_path: str,
    *,
    collection: str | None,
    out_format: str,
) -> None:
    settings = get_settings()

    def on_progress(query_id: str, backend: str, latency_ms: float) -> None:
        if out_format != "json":
            click.echo(f"  {query_id} / {backend}... {round(latency_ms)}ms", err=True)

    async with LlmClient(settings.llm_base_url) as llm_client:
        result = await run_benchmark(
            session, user, llm_client, settings, fixture_path,
            collection=collection, on_progress=on_progress,
        )

    if out_format == "json":
        click.echo(bench_result_to_json(result))
        return

    click.echo("\n" + format_bench_table(result.results))
    click.echo("Summary:")
    click.echo("-" * 70)
    click.echo(format_bench_summary(result.summary))
