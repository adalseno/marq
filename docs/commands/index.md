# Commands

Every marq command supports `--help`. The full auto-generated tree is at
[CLI reference](cli-reference.md); this section groups commands by what
they do and links to a deeper page for the ones worth explaining beyond
their `--help` text.

## Read-only

Safe to run freely — nothing here mutates the index.

| Command | Purpose |
|---|---|
| `search` | Full-text keyword search (BM25, no LLM) |
| `vsearch` | Vector similarity search (no reranking) |
| `query` / `deep-search` | Hybrid: expansion + RRF fusion + reranking (recommended) |
| `get` | Get a document by path or `#docid` |
| `multi-get` | Get multiple documents by glob or comma-separated list |
| `ls` | List collections, or files in a collection |
| `status` | Show index status and collections |
| `doctor` | Diagnose Postgres/pgvector/migration/LLM-router health |
| `bench` | Run search-quality benchmarks against a fixture file |

→ [Search & query](search-and-query.md) covers `search`/`vsearch`/`query`
in depth. → [Bench & doctor](bench-and-doctor.md) covers `bench` and
`doctor`.

## Mutating

Index-changing — collection and context management.

| Command | Purpose |
|---|---|
| `collection add/list/show/remove/rename/update-cmd/include/exclude` | Manage indexed collections |
| `context add/list/rm/check` | Manage per-path and global search context |
| `update` | Re-index all collections |
| `embed` | Generate vector embeddings for pending documents |
| `cleanup` | Remove orphaned content/embeddings/inactive documents |

→ [Collections & context](collections.md) covers all of these.

## Agent integration

| Command | Purpose |
|---|---|
| `mcp` / `mcp --http` / `mcp stop` | Start (or stop) the MCP server |
| `skill show` / `skill install` | Show or install the bundled agent skill |
| `skills list` / `get` / `path` | Discover bundled skills generically |

→ [MCP server](../mcp-server.md) covers `mcp`. → [Agent skills](../skills.md)
covers `skill`/`skills`.

## Output formats

`search`, `vsearch`, `query`, and `multi-get` all accept `--format`:

`cli` (default, human-readable) · `json` · `csv` · `md` · `xml` · `files`
· `toon` ([a compact LLM-oriented tabular encoding](https://toonformat.dev/))

See [Search & query › Output formats](search-and-query.md#output-formats)
for an example of each.
