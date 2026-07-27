"""Skill discovery/parsing - a "skill" is a directory with a SKILL.md
(YAML-ish frontmatter + markdown body) describing how an agent should use
qmd-py. Bundled skills live under `skills/bundled/` (package data,
resolved via `importlib.resources` so this works whether qmd-py runs
from a source checkout or an installed wheel) - port of the TS
reference's discoverSkills/findSkill/parseSkillFrontmatter
(src/cli/qmd.ts), scoped down: no external `QMD_SKILLS_DIR` search-path
override (qmd-py bundles exactly one skill today; that's a reasonable
place to extend later, not something to build ahead of need).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from importlib import resources
from pathlib import Path


@dataclass
class SkillInfo:
    name: str
    description: str
    dir: Path
    hidden: bool


_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---", re.DOTALL)
_CONTINUATION_RE = re.compile(r"^\s+\S")


def parse_skill_frontmatter(content: str) -> dict[str, object] | None:
    match = _FRONTMATTER_RE.match(content.lstrip())
    if not match:
        return None

    name = ""
    description_parts: list[str] = []
    hidden = False
    lines = match.group(1).splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("name:"):
            name = line[len("name:") :].strip()
        elif line.startswith("description:"):
            description_parts = [line[len("description:") :].strip()]
            while i + 1 < len(lines) and _CONTINUATION_RE.match(lines[i + 1]):
                i += 1
                description_parts.append(lines[i].strip())
        elif line.startswith("hidden:"):
            hidden = line[len("hidden:") :].strip().lower() in ("true", "yes")
        i += 1

    if not name:
        return None
    return {"name": name, "description": " ".join(description_parts), "hidden": hidden}


def _bundled_skills_dir() -> Path:
    return Path(str(resources.files("qmd_py.skills") / "bundled"))


def discover_skills() -> list[SkillInfo]:
    """Every bundled skill, hidden or not - callers that want the public
    listing (`skills list`/`skills get --all`) filter on `.hidden`
    themselves, matching the TS reference's discoverSkills()/
    runtimeSkills() split."""
    root = _bundled_skills_dir()
    if not root.is_dir():
        return []

    skills: list[SkillInfo] = []
    for entry in sorted(root.iterdir()):
        skill_path = entry / "SKILL.md"
        if not skill_path.is_file():
            continue
        parsed = parse_skill_frontmatter(skill_path.read_text(encoding="utf-8"))
        if parsed is None:
            continue
        skills.append(
            SkillInfo(
                name=str(parsed["name"]),
                description=str(parsed["description"]),
                dir=entry,
                hidden=bool(parsed["hidden"]),
            )
        )
    return sorted(skills, key=lambda s: s.name)


def find_skill(name: str) -> SkillInfo | None:
    return next((s for s in discover_skills() if s.name == name), None)


def read_skill_content(skill: SkillInfo) -> str:
    return (skill.dir / "SKILL.md").read_text(encoding="utf-8")


def collect_skill_files(skill: SkillInfo) -> list[tuple[str, str]]:
    """(relative_path, content) for every file under references/
    templates/scripts - supplementary material `skills get --full`
    includes alongside the SKILL.md body."""
    files: list[tuple[str, str]] = []
    for subdir_name in ("references", "templates", "scripts"):
        subdir = skill.dir / subdir_name
        if not subdir.is_dir():
            continue
        for entry in sorted(subdir.iterdir()):
            if entry.is_file():
                files.append((f"{subdir_name}/{entry.name}", entry.read_text(encoding="utf-8")))
    return files
