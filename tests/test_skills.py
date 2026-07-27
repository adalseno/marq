"""Phase 10: skill discovery/parsing unit tests (no DB/LLM needed)."""

from qmd_py.skills import (
    collect_skill_files,
    discover_skills,
    find_skill,
    parse_skill_frontmatter,
    read_skill_content,
)


def test_parse_skill_frontmatter_basic() -> None:
    content = "---\nname: foo\ndescription: does a thing\n---\n\nbody"
    parsed = parse_skill_frontmatter(content)
    assert parsed == {"name": "foo", "description": "does a thing", "hidden": False}


def test_parse_skill_frontmatter_multiline_description() -> None:
    content = "---\nname: foo\ndescription: line one\n  line two\nhidden: true\n---\n"
    parsed = parse_skill_frontmatter(content)
    assert parsed is not None
    assert parsed["description"] == "line one line two"
    assert parsed["hidden"] is True


def test_parse_skill_frontmatter_no_frontmatter_returns_none() -> None:
    assert parse_skill_frontmatter("# just a heading\n\nbody") is None


def test_parse_skill_frontmatter_missing_name_returns_none() -> None:
    content = "---\ndescription: no name here\n---\n"
    assert parse_skill_frontmatter(content) is None


def test_discover_skills_finds_bundled_marq_skill() -> None:
    skills = discover_skills()
    assert any(s.name == "marq" for s in skills)


def test_find_skill_by_name() -> None:
    skill = find_skill("marq")
    assert skill is not None
    assert skill.name == "marq"
    assert (skill.dir / "SKILL.md").is_file()


def test_find_skill_missing_returns_none() -> None:
    assert find_skill("does-not-exist") is None


def test_read_skill_content_matches_file() -> None:
    skill = find_skill("marq")
    assert skill is not None
    content = read_skill_content(skill)
    assert content.startswith("---")
    assert "name: marq" in content


def test_collect_skill_files_empty_for_bundled_skill() -> None:
    # The bundled marq skill has no references/templates/scripts subdirs.
    skill = find_skill("marq")
    assert skill is not None
    assert collect_skill_files(skill) == []
