# Collections & context

## Collections

Any folder becomes a searchable collection:

```console
$ uv run marq collection add tests/fixtures/sample-collection --name tasknote --mask '**/*.{md,py,js,ts}'
Creating collection 'tasknote'...
✓ Collection 'tasknote' created successfully
  Indexed: 6 new, 0 updated, 0 unchanged, 0 removed
```

`--mask` (default `**/*.md`) is a glob controlling which files get
indexed. `--name` defaults to the directory's basename.

```console
$ uv run marq collection show tasknote
Collection: tasknote
  Path:     /path/to/tests/fixtures/sample-collection
  Pattern:  **/*.{md,py,js,ts}
  Include:  yes (default)
```

### Content-addressing: why re-indexing is (almost always) free

Every document's body is stored once, keyed by its SHA-256 hash, in a
table shared across *every* collection — not per-collection. Indexing a
file whose content already exists anywhere in the database (a second
git worktree, a branch that's mostly the same as `main`, a file you
happened to index in two different collections) costs nothing beyond
recognizing the hash already exists. There's no separate "copy this
collection cheaply" feature because the storage design already makes
that free — just run `collection add` again on the new worktree/branch.
See [Architecture › Storage](../architecture.md#storage).

### Default-query inclusion

```console
$ uv run marq collection exclude tasknote
✓ Collection 'tasknote' excluded from default queries
$ uv run marq collection include tasknote
✓ Collection 'tasknote' included in default queries
```

`search`/`vsearch`/`query` search every `include`d collection when you
don't pass `-c`/`--collection`. Excluding a collection keeps it fully
searchable via an explicit `-c`, just not swept into unscoped queries —
useful for a large or noisy collection you only want to search
on purpose.

### Pre-update command

```console
$ uv run marq collection update-cmd tasknote -- git pull --ff-only
✓ Set update command for 'tasknote': git pull --ff-only
```

Runs before re-indexing whenever `marq update` touches this collection
— handy for a folder that's a git checkout you want kept current
automatically. Note the `--` before the command: without it, `click`
tries to parse `--ff-only` as an option of `update-cmd` itself, not part
of the command to store, and errors out. Clear it by passing no command:

```console
$ uv run marq collection update-cmd tasknote
✓ Cleared update command for 'tasknote'
```

### Other collection commands

`collection list` (all collections with details), `collection remove
<name>`, `collection rename <old> <new>` — all straightforward; see the
[CLI reference](cli-reference.md).

## Context

Context is a short, human-written note attached to a path (or globally)
that gets surfaced alongside search results — useful for disambiguating
what a document actually *is* when the content alone doesn't make that
obvious (a directory of raw data dumps, a folder of meeting transcripts,
etc.).

```console
$ uv run marq context add tasknote/src "Backend implementation: Python core + JS API layer"
✓ Set context for marq://tasknote/src

$ uv run marq context list
marq://tasknote/src
  Backend implementation: Python core + JS API layer
```

Pass `/` as the path to set global context — a note included for every
search, regardless of collection.

### `context check` — finding coverage gaps

```console
$ uv run marq context check
Collections without any context:
  tasknote (6 docs)
```

After adding context for `tasknote/src` above, re-running it reports the
collection is covered but flags the still-uncovered top-level path:

```console
$ uv run marq context check
Collections with uncovered top-level paths:
  tasknote: docs
```

Remove context the same way you added it:

```console
$ uv run marq context rm tasknote/src
✓ Removed context for marq://tasknote/src
```

## Indexing & maintenance

- **`marq update`** — re-indexes every collection (running each one's
  pre-update command first, if set).
- **`marq embed [-c <name>] [--model <slug>]`** — generates vector
  embeddings for documents that don't have them yet, for the configured
  (or specified) embedding model. Needed for `vsearch`/`query`, not
  `search`.
- **`marq cleanup`** — removes orphaned content/embeddings and inactive
  document records left behind by re-indexing or removing files. No
  `VACUUM` step (unlike the original qmd): Postgres's autovacuum handles
  space reclaim automatically, and a manual `VACUUM` on a shared server
  is more likely to cause lock contention than help.
