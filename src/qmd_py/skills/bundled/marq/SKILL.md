---
name: marq
description: Search local markdown/code knowledge bases indexed by marq (Postgres/pgvector). Use when users ask to find notes, retrieve documents, inspect a wiki, or answer from indexed local files.
license: MIT
compatibility: Requires the marq CLI or its MCP server. Run `marq skill show` for version-matched instructions.
allowed-tools: Bash(marq:*), mcp__marq__*
---

# marq - centralized markdown/code search

## How search works

marq searches collections indexed into Postgres/pgvector: notes, docs, wikis,
and project source. Use it before web search when the answer may already be in
an indexed local collection.

The workflow is always:

1. Search for candidate documents.
2. Retrieve the full source with `marq get` or `marq multi-get`.
3. Answer from retrieved text, citing paths or docids - not from snippets alone.

Typical loop:

```bash
marq search "connection pool timeout" -n 5
# leads: #abc123 notes/db-tuning.md; #def432 sources/incident-42.md
marq multi-get "#abc123,#def432" --format md
```

## Pick the right search mode

Use **`marq search`** (BM25, no LLM) when you know exact words, titles, names,
code symbols, or rare phrases:

```bash
marq search "cockpit OKR Goodhart" -n 10
marq search '"exact phrase"' -c notes -n 5
```

Use **`marq query`** (hybrid: expansion + BM25/vector RRF fusion + reranking)
as the default for anything conceptual, indirectly-described, or worded
differently than the source. **Prefer authoring the structured form yourself**
rather than passing a bare sentence and hoping the built-in expansion model
guesses right - you know the user's actual goal, domain vocabulary, and the
nearby-but-wrong concepts to avoid:

```bash
marq query "$(printf 'intent: find the incident writeup about connection pool exhaustion, not general perf tuning\nlex: connection pool timeout exhaustion -tuning\nvec: why database connections time out under high concurrency\nhyde: The incident occurred when all pooled connections were in use and new requests queued until the pool timeout fired.')"
```

Structured query fields (each line is optional except at least one of
lex/vec/hyde):

- `intent:` - what you are trying to find, and what to avoid. Steers ranking
  and snippet extraction. Always worth including.
- `lex:` - exact terms, aliases, code symbols, rare words you expect verbatim.
- `vec:` - a natural-language paraphrase of the idea.
- `hyde:` - a short passage describing what the answer would look like.

A bare `marq query "the user's sentence"` still works (auto-expanded), but
throws away context only you have - prefer the structured form when you can.

If you genuinely have nothing to expand (a single rare token, a verbatim
phrase), use `marq search` instead of `marq query`.

Add `--explain` (with `--format json` or the default `cli` format) to see the
RRF rank/weight and rerank score behind each result's final score.
Add `--no-rerank` for faster, lower-quality results on a CPU-only setup.

## Retrieve sources

Search results include a docid like `#abc123` and a `marq://collection/path`
virtual path. Fetch them:

```bash
marq get "#abc123"
marq get marq://notes/db-tuning.md
marq get notes/db-tuning.md:40:20      # 20 lines starting at line 40
marq multi-get "#abc123,#def432" --format md
marq multi-get 'notes/2025-*.md' -l 80
```

`get` and `multi-get` are line-numbered by default and always show the
document's `#docid` and `marq://` path:

```text
marq://notes/db-tuning.md  #abc123
---

1: # Connection pool tuning
2:
3: ...
```

Cite the docid and line numbers in your answer. Pass `--no-line-numbers` only
when you need raw content to reproduce verbatim (e.g. a code block).

## Output formats

`search`, `vsearch`, `query`, and `multi-get` all support `--format`:
`cli` (default, human-readable) | `json` | `csv` | `md` | `xml` | `files` |
`toon` (a compact LLM-oriented tabular encoding - see https://toonformat.dev/).
Prefer `--format json` or `--format toon` when you need to parse the output
programmatically rather than read it.

## Other commands

```bash
marq ls                          # list collections
marq ls <collection>[/path]      # list files in a collection
marq status                      # index health: doc counts, collections
marq doctor                      # diagnose Postgres/pgvector/LLM-router health
marq collection list/add/remove/rename/show/update-cmd/include/exclude
marq context add/list/rm/check   # per-path human-written context notes
marq update                      # re-index configured collections
marq embed                       # generate/refresh vector embeddings
marq mcp                         # start the MCP server (stdio, for agents/IDEs)
marq bench <fixture.json>        # search-quality benchmarks (precision/recall/MRR)
```

Never run `marq collection add`, `marq embed`, or `marq update` on the
user's behalf without being asked - these mutate the index. Read-only commands
(`search`, `vsearch`, `query`, `get`, `multi-get`, `ls`, `status`, `doctor`) are
safe to run freely.
