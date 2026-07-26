"""Pure HTTP client for the llama.cpp router (`QMD_LLM_BASE_URL`) - no local
model loading concept at all, unlike the TS reference's node-llama-cpp
based `src/llm.ts`. Prompt formatting is ported from that file's
`formatQueryForEmbedding`/`formatDocForEmbedding`/`isQwen3EmbeddingModel`
since those templates must match exactly for embeddings to be comparable
to the TS reference's.
"""

import re
from types import TracebackType
from typing import Self

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
    `/tokenize`, and `/detokenize` endpoints. `/v1/chat/completions` and
    `/rerank` are added in Phase 8 when query expansion/reranking land."""

    def __init__(self, base_url: str, timeout: float = 120.0) -> None:
        self._client = httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=timeout)

    async def embed(self, texts: list[str], model: str) -> list[list[float]]:
        response = await self._client.post(
            "/v1/embeddings", json={"model": model, "input": texts}
        )
        response.raise_for_status()
        data = response.json()["data"]
        return [item["embedding"] for item in sorted(data, key=lambda d: d["index"])]

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
