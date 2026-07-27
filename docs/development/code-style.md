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

**Google style** (`docstring_style = "google"` in `zensical.toml`), with
the narrative kept in front. `src/qmd_py/store/collection.py` is the
worked reference — copy its shape.

The primary content is still **decision-log prose**: what a typical
docstring explains is *why* a design choice was made, what alternative
was rejected and why, or what real bug shaped the current code. Google
sections are **additive**, not a replacement — they exist for the few
things prose states badly.

For example (`src/qmd_py/config.py`'s actual docstring on
`sqlalchemy_url`):

> Deliberately just our own schema, NOT `qmd_py,public`. TSVECTOR is a
> built-in Postgres type (doesn't need `public` on the path at all), and
> a multi-entry search_path breaks Alembic autogenerate in a subtle
> way: Postgres's `pg_table_is_visible()` ... considers a table visible
> if ANY search_path entry resolves to it — so with `qmd_py,public` on
> the path, [it] proposed dropping [unrelated tables it shouldn't have
> touched] ...

That preamble is the part that carries weight: a contributor reading a
signature can already see the parameter types (this project is `mypy
--strict` end to end); what they *can't* see is why it's shaped that way,
or what it would break to "simplify" it. When you make a non-obvious
choice, or fix a bug that wasn't obvious from the code alone, that's what
the docstring is for.

### Which sections to use

Three rules, in order of how much they matter:

1. **Never put types in the docstring.** Write `name: Unique per owner`,
   not `name (str): ...`. Everything is annotated and
   `show_signature_annotations` renders the real signature, so a type in
   prose is a second copy that mypy can't check and that drifts on the
   first refactor.
2. **`Raises:` almost always earns its place.** Which exception a caller
   gets — and when — is the least guessable thing about these functions.
   `CollectionNotFoundError` versus `PermissionDeniedError` depends on
   whether the lookup prefiltered on ownership in SQL; an `IntegrityError`
   on flush versus an up-front check is a real difference to the caller.
3. **`Args:` is optional; `Returns:` is worth it for the dataclasses.**
   `session`/`user` repeat on nearly every service function and
   documenting them each time is noise — skip them and describe only the
   arguments with something to say. `Returns:` earns its place where the
   type name doesn't tell the whole story (`ReindexResult`,
   `RemoveCollectionResult`), and `Attributes:` on the dataclass itself is
   usually the better home for that detail.

`Note:` is useful for the caveat that doesn't belong in the summary — a
known N+1, or a behaviour that was once wrong in an interesting way.

### Two things that will bite

**Section names are exact, and indentation matters.** `Args:` and
`Arguments:` both parse; `Params:` doesn't. A mis-indented section isn't
an error — griffe just stops recognising it and renders it as ordinary
prose, so it looks *almost* right. After writing your first few, build the
docs and look at the page.

**`zensical build --strict` is where malformed docstrings surface.** It
passes today. Keep it that way — a sloppy section can break the docs build
rather than degrading quietly.

The [Code reference](../reference/store.md) pages are generated from these
same docstrings, and show the annotated source alongside them
(`show_source = true` is intentional — see `zensical.toml`).

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
