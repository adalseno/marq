# Quickstart

This walks through installing marq, pointing it at a Postgres instance,
indexing a small collection, and running your first searches. It uses
the repo's own checked-in fixture
(`tests/fixtures/sample-collection/` — a tiny frozen "Tasknote" project
in markdown/Python/JavaScript/TypeScript) as the worked example, so you
can run every command below verbatim from a checkout and get the same
results.

## 1. Install

```bash
uv sync
```

This installs the `marq` console script into the project's virtualenv.
Use `uv run marq ...` below, or activate `.venv` and run `marq`
directly.

## 2. Start Postgres

```bash
cp .env.dev .env
podman-compose up -d
uv run alembic upgrade head
```

This starts a disposable `pgvector/pgvector:pg16` container on
`localhost:5433` (matching production's Postgres version) and applies
marq's schema migrations. See [Configuration](configuration.md) if
you'd rather point marq at your own Postgres instance instead.

!!! note "Docker works too — `podman-compose` is a preference, not a requirement"

    Substitute `docker compose up -d` and everything else is unchanged.
    The `postgres` service is plain Compose spec (environment, ports, a
    named volume, a healthcheck), and the fully-qualified
    `docker.io/pgvector/pgvector:pg16` image name — a podman idiom — is
    equally valid to Docker. The named volume also avoids the
    host-permission differences between rootless podman and rootful
    Docker.

    The caveats are all in the **optional** `llm` service, which is
    behind a profile and off by default: it passes through `/dev/dri` for
    GPU acceleration and labels its model mount `:Z` for SELinux. Both
    are Linux-host concepts, so that service won't start under Docker
    Desktop on macOS or Windows. Nothing else depends on it — see
    `llm-stack/README.md`.

You'll also need a reachable LLM endpoint for `vsearch`/`query`/`embed`
(anything OpenAI-endpoint-shaped for embeddings/chat/reranking) —
`.env.dev`'s default points at a real router; see
[Configuration](configuration.md#marq_llm_base_url) and
`llm-stack/README.md` if you want to run one locally instead.

## 3. Index a collection

```console
$ uv run marq collection add tests/fixtures/sample-collection --name tasknote --mask '**/*.{md,py,js,ts}'
Creating collection 'tasknote'...
✓ Collection 'tasknote' created successfully
  Indexed: 6 new, 0 updated, 0 unchanged, 0 removed
```

Any folder can become a collection. `--mask` is a glob pattern
controlling which files get indexed (default `**/*.md`); here it also
picks up the fixture's `.py`/`.js`/`.ts` files.

```console
$ uv run marq status
marq Status

Documents
  Total: 6 files indexed

Collections:
  tasknote (marq://tasknote/)
    Pattern:  **/*.{md,py,js,ts}
    Files:    6 (updated 2026-07-26)
```

## 4. Generate embeddings

```console
$ uv run marq embed
✓ Embedded 6 document(s), 6 chunk(s)
```

Needed for `vsearch` and `query` (semantic/hybrid search) — plain
`search` (BM25) works without it.

## 5. Search

```console
$ uv run marq search "priority" -c tasknote -n 3 --format files
#87b2c4,0.19,tasknote/src/tasks.py
#d8c0eb,0.17,tasknote/src/models.ts
#ff0a47,0.16,tasknote/src/api.js
```

```console
$ uv run marq query "how does the database schema get upgraded automatically" -c tasknote --intent "schema migrations" -n 1
---
# Changelog

**file:** `tasknote/CHANGELOG.md`
**docid:** `#3ba582`

@@ -4,4 @@ (3 before, 9 after)

- Added `due_date` to tasks, with an on-startup schema migration for
  existing databases.
- `GET /tasks?tag=` now filters by tag.
```

`query` is the recommended default: it expands your text into lexical/
semantic/hypothetical-document sub-queries, fuses the ranked lists with
Reciprocal Rank Fusion, and reranks the fused candidates with an LLM —
see [Search & query](commands/search-and-query.md) for the full picture,
including the structured `lex:`/`vec:`/`hyde:` query syntax for when you
want to control the sub-queries yourself.

## 6. Retrieve a document

```console
$ uv run marq get tasknote/CHANGELOG.md
marq://tasknote/CHANGELOG.md  #3ba582
---

1: # Changelog
2:
3: ## 0.3.0
4:
5: - Added `due_date` to tasks, with an on-startup schema migration for
6:   existing databases.
```

## Next steps

- [Configuration](configuration.md) — every `MARQ_*` setting.
- [Commands](commands/index.md) — the full CLI surface.
- [MCP server](mcp-server.md) — expose this same search over MCP to an
  agent or editor.

Clean up the example collection when you're done:

```bash
uv run marq collection remove tasknote
```
