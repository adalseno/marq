"""Pure HTTP client for the llama.cpp router (`MARQ_LLM_BASE_URL`) - no local
model loading concept at all, unlike the TS reference's node-llama-cpp
based `src/llm.ts`. Prompt formatting is ported from that file's
`formatQueryForEmbedding`/`formatDocForEmbedding`/`isQwen3EmbeddingModel`
since those templates must match exactly for embeddings to be comparable
to the TS reference's.
"""

import json
import re
from types import TracebackType
from typing import Any, Self

import httpx

_QWEN_EMBED_PATTERN = re.compile(r"qwen.*embed", re.IGNORECASE)
_EMBED_QWEN_PATTERN = re.compile(r"embed.*qwen", re.IGNORECASE)


def is_qwen3_embedding_model(model: str) -> bool:
    return bool(_QWEN_EMBED_PATTERN.search(model) or _EMBED_QWEN_PATTERN.search(model))


def format_query_for_embedding(query: str, model: str) -> str:
    """nomic-style task-prefix format (default) vs. Qwen3-Embedding's
    instruct format - the two embedding-prompt styles the TS reference
    supports."""
    if is_qwen3_embedding_model(model):
        return f"Instruct: Retrieve relevant documents for the given query\nQuery: {query}"
    return f"task: search result | query: {query}"


def format_doc_for_embedding(text: str, title: str | None, model: str) -> str:
    if is_qwen3_embedding_model(model):
        return f"{title}\n{text}" if title else text
    return f"title: {title or 'none'} | text: {text}"


class LlmClient:
    """Thin wrapper over the router's OpenAI-compatible `/v1/embeddings`,
    `/v1/chat/completions`, `/rerank`, `/tokenize`, and `/detokenize`
    endpoints."""

    def __init__(
        self,
        base_url: str,
        timeout: float = 120.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        """`transport` is httpx's standard injection seam, used by the
        tests to serve canned router responses through an
        `httpx.MockTransport` without a live router. Production callers
        leave it None and get httpx's real networking.
        """
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"), timeout=timeout, transport=transport
        )

    async def list_models(self) -> list[str]:
        """Model ids the router currently has presets for - backs `doctor`'s
        check that the configured embed/generate/rerank models actually
        exist on the router, not just in qmd-py's own settings."""
        response = await self._client.get("/v1/models")
        response.raise_for_status()
        return [item["id"] for item in response.json()["data"]]

    async def embed(self, texts: list[str], model: str) -> list[list[float]]:
        response = await self._client.post(
            "/v1/embeddings", json={"model": model, "input": texts}
        )
        response.raise_for_status()
        data = response.json()["data"]
        return [item["embedding"] for item in sorted(data, key=lambda d: d["index"])]

    async def chat_json(
        self,
        messages: list[dict[str, str]],
        model: str,
        json_schema: dict[str, Any],
        max_tokens: int = 300,
        temperature: float = 0.7,
    ) -> dict[str, Any]:
        """One chat completion constrained to `json_schema` via the
        router's `response_format` (an OpenAI-compatible, increasingly
        standard llama.cpp server feature) - returns the parsed JSON
        object. Used for query expansion (see search/hybrid.py)."""
        response = await self._client.post(
            "/v1/chat/completions",
            json={
                "model": model,
                "messages": messages,
                "response_format": {"type": "json_schema", "json_schema": json_schema},
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
        )
        response.raise_for_status()
        content: dict[str, Any] = json.loads(response.json()["choices"][0]["message"]["content"])
        return content

    async def rerank(self, query: str, documents: list[str], model: str) -> list[float]:
        """Relevance scores in the same order as `documents` - the
        router's response is keyed by `index` rather than guaranteed to
        preserve input order, so results are re-sorted into place here.

        Always returns one score per document (unscored ones stay 0.0);
        callers rely on that length, hybrid.py zipping it against its
        candidate chunks with `strict=True`. Out-of-range indices are
        dropped rather than assigned: a stray large index would raise
        IndexError, and - worse, because it fails silently - a negative
        one would write onto the wrong document from the end.
        """
        response = await self._client.post(
            "/rerank", json={"model": model, "query": query, "documents": documents}
        )
        response.raise_for_status()
        results = response.json()["results"]
        scores = [0.0] * len(documents)
        for r in results:
            index = r["index"]
            if 0 <= index < len(scores):
                scores[index] = r["relevance_score"]
        return scores

    async def tokenize(self, text: str, model: str) -> list[int]:
        response = await self._client.post("/tokenize", json={"model": model, "content": text})
        response.raise_for_status()
        tokens: list[int] = response.json()["tokens"]
        return tokens

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.aclose()
