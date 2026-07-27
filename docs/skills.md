# Agent skills

marq bundles an [Agent Skill](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills)
— a small markdown file with frontmatter that teaches an agent (Claude
Code or similar) how to use marq effectively: when to reach for
`search` vs. `query`, how to author structured `lex:`/`vec:`/`hyde:`
queries, how to cite results, and which commands are safe to run freely
vs. which mutate the index.

## Show and install

```console
$ marq skill show
---
name: marq
description: Search local markdown/code knowledge bases indexed by marq (Postgres/pgvector). Use when users ask to find notes, retrieve documents, inspect a wiki, or answer from indexed local files.
license: MIT
compatibility: Requires the marq CLI or its MCP server. Run `marq skill show` for version-matched instructions.
allowed-tools: Bash(marq:*), mcp__marq__*
---
...
```

```console
$ marq skill install
✓ Installed marq skill to /path/to/project/.agents/skills/marq
```

`skill install` copies the bundled skill into `.agents/skills/marq`
(under the current directory) or, with `--global`,
`~/.agents/skills/marq`. `--force` overwrites an existing install.

The *installed* copy is intentionally a small bootstrap stub — a bang
command (`` !`marq skill show` ``) that loads the real, version-matched
instructions from whatever marq is actually installed, rather than a
frozen copy that could drift out of sync as marq changes. If your agent
doesn't support bang-command expansion, the stub tells it to run
`marq skill show` directly instead.

## Generic skill discovery

```console
$ marq skills list
  marq  Search local markdown/code knowledge bases indexed by marq (Postgres/pgvector)...

$ marq skills get marq
$ marq skills get marq --full     # include references/templates/scripts subdirs
$ marq skills path marq           # print the bundled skill's directory
```

`skills list/get/path` are a generic mechanism for discovering *any*
bundled skill, not specific to the one marq ships today — `skill
show`/`skill install` above are really just `skills get marq` and a
copy-with-a-stub-substitution built on top of the same discovery layer.

## How bundling works

The bundled skill lives at `src/qmd_py/skills/bundled/marq/SKILL.md`,
resolved at runtime via `importlib.resources` — this is what makes
`skill show`/`skills get` work identically whether marq is running from
a source checkout or an installed wheel, without hardcoding a filesystem
path that would only exist in one of those two cases.

A skill's `SKILL.md` is YAML-ish frontmatter (`name`, `description`,
optional `hidden: true` to exclude it from `skills list`/`get --all`
while still being reachable by exact name) followed by the markdown
body. `description` supports YAML-style line continuation (an indented
continuation line joins onto the previous one), matching how the
frontmatter parser in `src/qmd_py/skills/__init__.py` reads it.
