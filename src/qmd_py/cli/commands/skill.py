"""`skill show`/`skill install` (the bundled qmdpy skill) and `skills
list/get/path` (generic bundled-skill discovery) - port of the TS
reference's skill/skills CLI surface (src/cli/qmd.ts), scoped down: no
Claude-symlink auto-detection/prompt flow on install (that's optional UX
sugar, not core behavior) and no `QMD_SKILLS_DIR` override (see
skills/__init__.py's module docstring).
"""

import json
import shutil
from pathlib import Path

import click

from qmd_py.skills import (
    SkillInfo,
    collect_skill_files,
    discover_skills,
    find_skill,
    read_skill_content,
)

_SKILL_NAME = "qmdpy"


def _install_dir(global_install: bool) -> Path:
    base = Path.home() if global_install else Path.cwd()
    return base / ".agents" / "skills" / _SKILL_NAME


def _installed_stub_content() -> str:
    return f"""---
name: {_SKILL_NAME}
description: Bootstrap qmd-py search instructions from the installed qmdpy CLI.
  Use when users ask to find or retrieve notes, docs, or indexed local markdown.
license: MIT
compatibility: Requires the qmdpy CLI. Run `qmdpy skill show` for version-matched instructions.
allowed-tools: Bash(qmdpy:*), mcp__qmd__*
---

# qmd-py - Query Markdown Documents

This installed skill is intentionally a small bootstrap so it does not go
stale when the qmd-py package updates.

Load the full, version-matched instructions from the CLI:

!`qmdpy skill show`

If your agent does not support bang-command expansion, run:

```bash
qmdpy skill show
```

Then follow those instructions. In short: search first, fetch full sources
with `qmdpy get` or `qmdpy multi-get`, and answer from retrieved text rather
than snippets.
"""


@click.command("skill")
@click.argument("action", type=click.Choice(["show", "install"]))
@click.option("--global", "global_install", is_flag=True, help="Install to ~/.agents/skills")
@click.option("--force", is_flag=True, help="Overwrite an existing install")
def skill_command(action: str, global_install: bool, force: bool) -> None:
    """Show or install the qmd-py skill (bootstrap instructions for agents)."""
    skill = find_skill(_SKILL_NAME)
    if skill is None:
        raise click.ClickException("qmdpy skill not found (package installation issue).")

    if action == "show":
        content = read_skill_content(skill)
        click.echo(content if content.endswith("\n") else content + "\n", nl=False)
        return

    target_dir = _install_dir(global_install)
    if target_dir.exists():
        if not force:
            raise click.ClickException(
                f"Skill already exists: {target_dir} (use --force to replace it)"
            )
        shutil.rmtree(target_dir)

    shutil.copytree(skill.dir, target_dir)
    (target_dir / "SKILL.md").write_text(_installed_stub_content(), encoding="utf-8")
    click.echo(f"✓ Installed qmdpy skill to {target_dir}")


def _output_json(payload: object) -> None:
    click.echo(json.dumps(payload))


@click.group("skills", invoke_without_command=True)
@click.pass_context
def skills_group(ctx: click.Context) -> None:
    """List and retrieve bundled runtime skills."""
    if ctx.invoked_subcommand is None:
        ctx.invoke(list_command)


def _runtime_skills() -> list[SkillInfo]:
    return [s for s in discover_skills() if not s.hidden]


@skills_group.command("list")
@click.option("--json", "json_mode", is_flag=True, help="Output as JSON")
def list_command(json_mode: bool) -> None:
    """List bundled runtime skills."""
    skills = _runtime_skills()
    if json_mode:
        _output_json(
            {
                "success": True,
                "data": [{"name": s.name, "description": s.description} for s in skills],
            }
        )
        return
    if not skills:
        click.echo("No skills found")
        return
    max_name = max(len(s.name) for s in skills)
    for s in skills:
        click.echo(f"  {s.name.ljust(max_name)}  {s.description}")


@skills_group.command("get")
@click.argument("names", nargs=-1)
@click.option("--full", is_flag=True, help="Include references/templates/scripts")
@click.option("--all", "get_all", is_flag=True, help="Print all bundled runtime skills")
@click.option("--json", "json_mode", is_flag=True, help="Output as JSON")
def get_command(names: tuple[str, ...], full: bool, get_all: bool, json_mode: bool) -> None:
    """Print one or more bundled runtime skills."""
    if get_all:
        targets = _runtime_skills()
    else:
        targets = []
        for name in names:
            skill = find_skill(name)
            if skill is None:
                raise click.ClickException(f"Skill not found: {name}")
            targets.append(skill)

    if not targets:
        raise click.ClickException("No skill name provided. Usage: qmdpy skills get <name>")

    if json_mode:
        data = []
        for s in targets:
            entry: dict[str, object] = {"name": s.name, "content": read_skill_content(s)}
            if full:
                entry["files"] = [
                    {"path": path, "content": content} for path, content in collect_skill_files(s)
                ]
            data.append(entry)
        _output_json({"success": True, "data": data})
        return

    for i, s in enumerate(targets):
        if i > 0:
            click.echo("\n---\n")
        content = read_skill_content(s)
        click.echo(content if content.endswith("\n") else content + "\n", nl=False)
        if full:
            for path, content in collect_skill_files(s):
                click.echo(f"\n--- {path} ---\n")
                click.echo(content if content.endswith("\n") else content + "\n", nl=False)


@skills_group.command("path")
@click.argument("name", required=False)
@click.option("--json", "json_mode", is_flag=True, help="Output as JSON")
def path_command(name: str | None, json_mode: bool) -> None:
    """Print a bundled skill's directory (or all bundled skill directories)."""
    if not name:
        paths = [str(s.dir) for s in _runtime_skills()]
        if json_mode:
            _output_json({"success": True, "data": {"paths": paths}})
        else:
            for p in paths:
                click.echo(p)
        return

    skill = find_skill(name)
    if skill is None:
        raise click.ClickException(f"Skill not found: {name}")
    if json_mode:
        _output_json({"success": True, "data": {"name": skill.name, "path": str(skill.dir)}})
    else:
        click.echo(str(skill.dir))
