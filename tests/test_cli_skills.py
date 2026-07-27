"""`marq skill` / `marq skills` CLI tests.

Unlike the rest of the CLI surface these touch neither Postgres nor the
LLM router - they only read the bundled skill files and write to an
isolated filesystem - so they run in the default (`-m "not integration"`)
suite.
"""

import json
from pathlib import Path

import pytest
from click.testing import CliRunner, Result

from qmd_py.cli.main import cli


def _run(*args: str) -> Result:
    return CliRunner().invoke(cli, list(args))


def test_skill_show_prints_bundled_instructions() -> None:
    result = _run("skill", "show")

    assert result.exit_code == 0, result.output
    assert "marq" in result.output


def test_skill_show_output_ends_with_newline() -> None:
    """The command normalizes a missing trailing newline rather than
    letting the shell prompt run into the last line."""
    result = _run("skill", "show")

    assert result.output.endswith("\n")


def test_skill_install_writes_bootstrap_stub() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem() as cwd:
        result = runner.invoke(cli, ["skill", "install"])

        assert result.exit_code == 0, result.output
        installed = Path(cwd) / ".agents" / "skills" / "marq" / "SKILL.md"
        assert installed.exists()
        content = installed.read_text(encoding="utf-8")
        assert "marq skill show" in content
        assert "name: marq" in content


def test_skill_install_refuses_to_overwrite_without_force() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        runner.invoke(cli, ["skill", "install"])

        result = runner.invoke(cli, ["skill", "install"])

        assert result.exit_code != 0
        assert "already exists" in result.output


def test_skill_install_force_replaces_existing() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem() as cwd:
        runner.invoke(cli, ["skill", "install"])
        stale = Path(cwd) / ".agents" / "skills" / "marq" / "stale.md"
        stale.write_text("stale", encoding="utf-8")

        result = runner.invoke(cli, ["skill", "install", "--force"])

        assert result.exit_code == 0, result.output
        assert not stale.exists()


def test_skill_install_global_targets_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--global writes under ~/.agents/skills; Path.home is redirected so
    the test never touches the real home directory."""
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))

    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(cli, ["skill", "install", "--global"])

    assert result.exit_code == 0, result.output
    assert (tmp_path / ".agents" / "skills" / "marq" / "SKILL.md").exists()


def test_skills_list_names_the_bundled_skill() -> None:
    result = _run("skills", "list")

    assert result.exit_code == 0, result.output
    assert "marq" in result.output


def test_skills_group_without_subcommand_lists() -> None:
    bare = _run("skills")
    listed = _run("skills", "list")

    assert bare.output == listed.output


def test_skills_list_json_is_machine_readable() -> None:
    result = _run("skills", "list", "--json")

    payload = json.loads(result.output)
    assert payload["success"] is True
    assert "marq" in [s["name"] for s in payload["data"]]


def test_skills_get_prints_skill_content() -> None:
    result = _run("skills", "get", "marq")

    assert result.exit_code == 0, result.output
    assert "marq" in result.output


def test_skills_get_unknown_name_fails() -> None:
    result = _run("skills", "get", "no-such-skill")

    assert result.exit_code != 0
    assert "Skill not found: no-such-skill" in result.output


def test_skills_get_without_name_explains_usage() -> None:
    result = _run("skills", "get")

    assert result.exit_code != 0
    assert "No skill name provided" in result.output


def test_skills_get_json_full_includes_files() -> None:
    result = _run("skills", "get", "marq", "--json", "--full")

    payload = json.loads(result.output)
    entry = payload["data"][0]
    assert entry["name"] == "marq"
    assert "files" in entry


def test_skills_get_all_prints_every_skill() -> None:
    result = _run("skills", "get", "--all")

    assert result.exit_code == 0, result.output
    assert result.output.strip()


def test_skills_path_prints_directory() -> None:
    result = _run("skills", "path", "marq")

    assert result.exit_code == 0, result.output
    assert Path(result.output.strip()).is_dir()


def test_skills_path_without_name_lists_all() -> None:
    result = _run("skills", "path")

    assert result.exit_code == 0, result.output
    assert result.output.strip()


def test_skills_path_json_reports_name_and_path() -> None:
    result = _run("skills", "path", "marq", "--json")

    payload = json.loads(result.output)
    assert payload["data"]["name"] == "marq"
    assert Path(payload["data"]["path"]).is_dir()


def test_skills_path_unknown_name_fails() -> None:
    result = _run("skills", "path", "no-such-skill")

    assert result.exit_code != 0
    assert "Skill not found" in result.output
