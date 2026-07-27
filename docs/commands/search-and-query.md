# Search & query

marq has three search commands, in increasing order of cost and quality:

| Command | Signal | LLM involved? | When to use |
|---|---|---|---|
| `search` | BM25 keyword (`tsvector`/`ts_rank`) | No | You know exact words, names, or rare phrases |
| `vsearch` | Vector similarity (pgvector/HNSW) | Embeddings only | Conceptual/semantic recall, no reranking |
| `query` (`deep-search`) | Both, fused + reranked | Yes | Best quality — the recommended default |

## `search` — full-text keyword search

Fast, exact, no LLM round-trip. Supports quoted phrases and
`-negation`:

```console
$ uv run marq search '"exact phrase"' -c notes -n 5
$ uv run marq search 'timeout -redis' -c notes -n 5
```

## `vsearch` — vector similarity

Embeds your query and finds the nearest chunks by cosine distance, no
reranking pass. Useful when you want semantic recall without paying for
an LLM rerank call:

```console
$ uv run marq vsearch "how does the rate limiter handle bursts" -c notes
```

## `query` — hybrid search (recommended)

`query` (alias `deep-search`) is the full pipeline: it expands your text
into typed sub-queries (lexical, semantic, hypothetical-document),
searches with each, fuses the ranked lists with Reciprocal Rank Fusion,
and reranks the fused candidates with an LLM.

```console
$ uv run marq query "priority levels" -c tasknote --explain -n 2
---
# Changelog

**file:** `tasknote/CHANGELOG.md`
**docid:** `#3ba582`

@@ -10,4 @@ (9 before, 3 after)

- Added priority levels (low/medium/high) to tasks.

## 0.1.0

---
# models

**file:** `tasknote/src/models.ts`
**docid:** `#d8c0eb`

@@ -4,4 @@ (3 before, 25 after)

export type Priority = "low" | "medium" | "high";

export interface Tag {


[tasknote/CHANGELOG.md] rrf(rank=1, weight=0.75, score=1.0000)  rerank=0.9783  blended=0.9946
[tasknote/src/models.ts] rrf(rank=3, weight=0.75, score=0.3333)  rerank=0.9293  blended=0.4823
```

`--explain` (shown above, `cli`/`json` formats only) shows, per result:
the RRF rank and weight it was fused at, the raw RRF position score, the
LLM rerank score, and the final blended score — `blended_score =
rrf_weight * rrf_position_score + (1 - rrf_weight) * rerank_score`,
where `rrf_weight` is `0.75`/`0.60`/`0.40` depending on how high the
candidate ranked before reranking (higher-ranked candidates trust RRF
more; lower-ranked ones lean on the reranker to have a chance at
recovery).

Other flags:

- `--intent TEXT` — background context to disambiguate the query and
  sharpen snippet extraction. Doesn't search on its own.
- `--no-rerank` — skip the LLM reranking pass, return RRF-only scores.
  Faster, lower quality; useful on a CPU-only setup.
- `-C, --candidate-limit N` (default 40) — how many RRF-fused candidates
  get reranked. Lower is faster but may miss results.

### Structured queries

Instead of a single string, `query` also accepts a multi-line query
document where every line is typed `lex:`, `vec:`, `hyde:`, or an
optional `intent:`. This bypasses automatic expansion entirely — you
supply the sub-queries yourself:

```console
$ uv run marq query "$(printf 'lex: due date\nvec: schema changes\nintent: understand database schema evolution')" -c tasknote --format files -n 3
#3ba582,1.00,tasknote/CHANGELOG.md
#87b2c4,0.49,tasknote/src/tasks.py
#d5e095,0.39,tasknote/README.md
```

- `lex:` — exact terms, aliases, code symbols, rare words you expect
  verbatim (same syntax as `search`: quoted phrases, `-negation`).
- `vec:` — a natural-language paraphrase of the idea.
- `hyde:` — a short passage describing what the answer would look
  like (a "hypothetical document").
- `intent:` — optional, same role as `--intent`.

The first sub-query (the implicit expansion, or the first typed line in
a structured document) gets 2x RRF weight relative to the rest — put
your strongest signal first.

## Output formats

`search`, `vsearch`, `query`, and `multi-get` all accept `--format`.
`cli` (the default, shown above) falls back to a markdown-flavored
render; here's the same result in the others:

=== "json"

    ```console
    $ uv run marq search "priority" -c tasknote -n 1 --format json
    [
      {
        "docid": "#87b2c4",
        "score": 0.19,
        "file": "tasknote/src/tasks.py",
        "line": 17,
        "title": "tasks",
        "snippet": "@@ -16,4 @@ (15 before, 55 after)\n    tag: str | None\n    priority: str\n    done: bool\n    due_date: date | None"
      }
    ]
    ```

=== "csv"

    ```console
    $ uv run marq search "priority" -c tasknote -n 1 --format csv
    docid,score,file,title,context,line,snippet
    #87b2c4,0.1868,tasknote/src/tasks.py,tasks,,17,"@@ -16,4 @@ (15 before, 55 after)
        tag: str | None
        priority: str
        done: bool
        due_date: date | None"
    ```

=== "toon"

    ```console
    $ uv run marq search "priority" -c tasknote -n 1 --format toon
    results[1]{docid,score,file,title,context,line,snippet}:
      #87b2c4,0.19,tasknote/src/tasks.py,tasks,"",17,"@@ -16,4 @@ (15 before, 55 after)\n    tag: str | None\n    priority: str\n    done: bool\n    due_date: date | None"
    ```

    [TOON](https://toonformat.dev/) is a compact, LLM-oriented tabular
    encoding — a marq-only addition (not in the original qmd), since
    cutting prompt-token overhead matters for a tool whose primary
    consumer is often an LLM agent, not a human terminal.

=== "files"

    ```console
    $ uv run marq search "priority" -c tasknote -n 3 --format files
    #87b2c4,0.19,tasknote/src/tasks.py
    #d8c0eb,0.17,tasknote/src/models.ts
    #ff0a47,0.16,tasknote/src/api.js
    ```

`md` and `xml` follow the same shape as `cli`/`json` respectively, just
in their own markup.
