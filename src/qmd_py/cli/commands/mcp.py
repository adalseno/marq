"""`mcp` / `mcp stop` - MCP server transports (stdio by default, `--http`
for Streamable HTTP, `--daemon` to background the HTTP server) - port of
the TS reference's `mcp` CLI surface (src/cli/qmd.ts).
"""

import os
import signal
import subprocess
import sys
from pathlib import Path

import click

from qmd_py.config import get_settings

_STATE_DIR = Path.home() / ".cache" / "marq"
_PID_FILE = _STATE_DIR / "mcp.pid"
_LOG_FILE = _STATE_DIR / "marq.log"
"""Where the daemon's own logging goes, via log.py's RotatingFileHandler
(size-capped, 3 backups). It replaces the raw, unbounded `mcp.log` the
daemon used to append stdout/stderr to forever - a disk problem for a
server meant to run indefinitely."""

_STDIO_FILE = _STATE_DIR / "mcp-stdio.log"
"""Anything the child writes outside the logging system (a traceback on
startup, a library printing to stderr) still has to land somewhere, so
Popen's stdout/stderr go here. Truncated at each daemon start rather
than appended, which is what made the old file grow without bound."""


def _process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _run_stdio() -> None:
    import anyio

    from qmd_py.mcp.server import create_mcp_server

    async def main() -> None:
        server = await create_mcp_server()
        await server.run_stdio_async()

    anyio.run(main)


def _run_http(host: str, port: int) -> None:
    import anyio

    from qmd_py.mcp.server import create_mcp_server

    async def main() -> None:
        server = await create_mcp_server(http=True, host=host, port=port)
        click.echo(f"marq MCP server listening on http://{host}:{port}/mcp", err=True)
        await server.run_streamable_http_async()

    anyio.run(main)


def _start_daemon(host: str, port: int) -> None:
    _STATE_DIR.mkdir(parents=True, exist_ok=True)
    if _PID_FILE.exists():
        existing_pid = int(_PID_FILE.read_text().strip())
        if _process_alive(existing_pid):
            click.echo(f"MCP daemon already running (pid {existing_pid})", err=True)
            raise SystemExit(1)
        _PID_FILE.unlink()

    settings = get_settings()
    # The daemon logs to a rotating file rather than the caller's stderr,
    # which is gone the moment this command returns. An explicit
    # MARQ_LOG_FILE still wins; MARQ_LOG_LEVEL defaults to INFO here
    # (one line per unit of work) rather than the CLI's WARNING, since
    # nobody is watching a background process interactively.
    # model_fields_set holds the settings actually provided via env/.env,
    # which is how an explicit MARQ_LOG_LEVEL=WARNING is told apart from
    # the default that the daemon overrides.
    daemon_level = settings.log_level if "log_level" in settings.model_fields_set else "INFO"
    env = {
        **os.environ,
        "MARQ_LOG_FILE": str(settings.log_file or _LOG_FILE),
        "MARQ_LOG_LEVEL": daemon_level,
    }
    with _STDIO_FILE.open("wb") as stdio_log:
        proc = subprocess.Popen(
            [
                sys.executable, "-m", "qmd_py.cli.main", "mcp",
                "--http", "--host", host, "--port", str(port),
            ],
            stdout=stdio_log,
            stderr=stdio_log,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            env=env,
        )
    _PID_FILE.write_text(str(proc.pid))
    click.echo(f"MCP daemon started (pid {proc.pid}), listening on http://{host}:{port}/mcp")
    click.echo(f"Logs: {env['MARQ_LOG_FILE']} (level {daemon_level})")


_LOOPBACK_HOSTS = frozenset({"localhost", "::1"})


def _warn_if_public_bind(host: str) -> None:
    """The HTTP transport has no authentication and `can_access()` is
    mocked to allow everything, so a non-loopback bind hands every
    indexed document to anyone who can reach the port. Warn rather than
    refuse - exposing it may be deliberate on a trusted network - until
    real ACL (and with it a bearer-token check) lands."""
    if host in _LOOPBACK_HOSTS or host.startswith("127."):
        return
    click.echo(
        f"Warning: binding {host} exposes the whole index over plain HTTP with NO "
        "authentication - anyone who can reach the port can read every indexed "
        "document. Use --host 127.0.0.1 unless this is a trusted network.",
        err=True,
    )


@click.group("mcp", invoke_without_command=True)
@click.option("--http", is_flag=True, help="Use Streamable HTTP transport instead of stdio")
@click.option("--port", type=int, default=8181, show_default=True, help="HTTP port")
@click.option("--host", default="127.0.0.1", show_default=True, help="HTTP host")
@click.option("--daemon", is_flag=True, help="Run the HTTP server in the background")
@click.pass_context
def mcp_group(ctx: click.Context, http: bool, port: int, host: str, daemon: bool) -> None:
    """Start the MCP server (stdio by default), or manage the HTTP daemon (see 'mcp stop')."""
    if ctx.invoked_subcommand is not None:
        return
    if daemon and not http:
        raise click.UsageError("--daemon requires --http")
    if http:
        _warn_if_public_bind(host)
    if daemon:
        _start_daemon(host, port)
    elif http:
        _run_http(host, port)
    else:
        _run_stdio()


@mcp_group.command("stop")
def stop_command() -> None:
    """Stop the background MCP HTTP daemon."""
    if not _PID_FILE.exists():
        click.echo("No MCP daemon is running.", err=True)
        raise SystemExit(1)
    pid = int(_PID_FILE.read_text().strip())
    if not _process_alive(pid):
        _PID_FILE.unlink()
        click.echo("No MCP daemon is running (stale pidfile removed).", err=True)
        raise SystemExit(1)
    os.kill(pid, signal.SIGTERM)
    _PID_FILE.unlink()
    click.echo(f"Stopped MCP daemon (pid {pid}).")
