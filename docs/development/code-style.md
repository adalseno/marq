# Code style & conventions

```bash
uv run ruff check --fix .      # lint (line length 100, E/F/I/UP/B rules)
uv run mypy src alembic tests  # strict type checking
uv run alembic check           # verify SQLModel models match the latest migration
```

All three are part of what "done" means for a change here, alongside
the live-verification discipline in [Setup & testing](testing.md) —
`ruff`/`mypy` catch what they catch, but neither proves a feature
actually works against real infrastructure.

## Docstring convention

This codebase's docstrings are **narrative, decision-log prose**, not
formal `Args:`/`Returns:`/`Examples:` API documentation. A typical
module docstring explains *why* a design choice was made, what
alternative was rejected and why, or what real bug shaped the current
shape of the code — not a parameter-by-parameter contract.

For example (`src/qmd_py/config.py`'s actual docstring on
`sqlalchemy_url`):

> Deliberately just our own schema, NOT `qmd_py,public`. TSVECTOR is a
> built-in Postgres type (doesn't need `public` on the path at all), and
> a multi-entry search_path breaks Alembic autogenerate in a subtle
> way: Postgres's `pg_table_is_visible()` ... considers a table visible
> if ANY search_path entry resolves to it — so with `qmd_py,public` on
> the path, [it] proposed dropping [unrelated tables it shouldn't have
> touched] ...

This is deliberate, not an oversight: a contributor reading a function
signature can already see its parameter types (this project is `mypy
--strict` end to end); what they *can't* see from the signature is why
it's shaped the way it is, or what it would break to "simplify" it.
Write new docstrings the same way — when you make a non-obvious choice,
or fix a bug that wasn't obvious from the code alone, that's what the
docstring is for.

This is also why the [Code reference](../reference/store.md) pages
(generated from these same docstrings via `mkdocstrings`) read the way
they do: expect a design-rationale preamble plus the annotated source
(`show_source = true` is intentional there, see `zensical.toml`), not a
parameter table. If you're used to numpy/Google-style API docs, don't
be surprised that these look different — it's the same tradeoff in
reverse: optimized for *why*, not for a parameter reference mypy already
gives you for free.

## Adding a new CLI command

- One file per command (or command group) under
  `src/qmd_py/cli/commands/`, following the existing pattern (a
  `click.command`/`click.group`, an `_impl` async function it delegates
  to via `cli/runtime.py`'s session-opening helper).
- Register it in `src/qmd_py/cli/main.py`.
- If it returns multiple results in a list, consider reusing
  `cli/formatter.py`'s existing `--format` machinery rather than adding
  a new one-off output path.
- User-facing strings (help text, error messages) should say `marq`,
  not `qmd` — see [Architecture › Naming](../architecture.md#naming) for
  why that distinction is intentional and maintained deliberately, not
  just cosmetic.
