"""LlmClient tests against canned router responses (`httpx.MockTransport`).

No Postgres, no live router - these run in the default
(`-m "not integration"`) suite. They cover the contracts the real router
only exercises when it happens to behave a certain way: index-keyed
response reordering, and what each method raises when the router errors,
times out, or returns something unparseable.

That error behavior is load-bearing rather than incidental:
search/hybrid.py's `expand_query()` catches exactly
(httpx.HTTPError, KeyError, ValueError, TypeError) to fall back to an
unexpanded query, so these tests pin the exception types that contract
depends on.
"""

import json
from collections.abc import Callable

import httpx
import pytest

from qmd_py.llm.client import (
    LlmClient,
    format_doc_for_embedding,
    format_query_for_embedding,
    is_qwen3_embedding_model,
)

Handler = Callable[[httpx.Request], httpx.Response]


def _client(handler: Handler) -> LlmClient:
    return LlmClient("http://router.invalid", transport=httpx.MockTransport(handler))


def _json_route(payload: object, status_code: int = 200) -> Handler:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, json=payload)

    return handler


# =============================================================================
# Embedding prompt formatting (pure)
# =============================================================================


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("qwen3-embedding-0.6b", True),
        ("Qwen3-Embedding-8B-Q8_0", True),
        ("embedding-qwen-custom", True),
        ("bge-m3-q8_0", False),
        ("nomic-embed-text", False),
    ],
)
def test_is_qwen3_embedding_model(model: str, expected: bool) -> None:
    assert is_qwen3_embedding_model(model) is expected


def test_format_query_uses_instruct_style_for_qwen() -> None:
    formatted = format_query_for_embedding("how does auth work", "qwen3-embedding-0.6b")

    assert formatted.startswith("Instruct: Retrieve relevant documents")
    assert formatted.endswith("Query: how does auth work")


def test_format_query_uses_task_prefix_by_default() -> None:
    assert format_query_for_embedding("auth", "bge-m3-q8_0") == "task: search result | query: auth"


def test_format_doc_uses_bare_title_body_for_qwen() -> None:
    assert format_doc_for_embedding("body", "Title", "qwen3-embedding-0.6b") == "Title\nbody"


def test_format_doc_without_title_for_qwen_is_body_only() -> None:
    assert format_doc_for_embedding("body", None, "qwen3-embedding-0.6b") == "body"


def test_format_doc_uses_labelled_fields_by_default() -> None:
    assert format_doc_for_embedding("body", "Title", "bge-m3") == "title: Title | text: body"


def test_format_doc_without_title_says_none() -> None:
    assert format_doc_for_embedding("body", None, "bge-m3") == "title: none | text: body"


# =============================================================================
# embed
# =============================================================================


async def test_embed_restores_input_order_from_index() -> None:
    """The router may answer out of order; `embed` must return vectors
    positionally matching its input, or every chunk gets the wrong
    embedding."""
    payload = {
        "data": [
            {"index": 2, "embedding": [0.3]},
            {"index": 0, "embedding": [0.1]},
            {"index": 1, "embedding": [0.2]},
        ]
    }
    async with _client(_json_route(payload)) as client:
        vectors = await client.embed(["a", "b", "c"], "bge-m3")

    assert vectors == [[0.1], [0.2], [0.3]]


async def test_embed_sends_model_and_inputs() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        assert request.url.path == "/v1/embeddings"
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [0.0]}]})

    async with _client(handler) as client:
        await client.embed(["only"], "bge-m3")

    assert seen == {"model": "bge-m3", "input": ["only"]}


async def test_embed_raises_on_server_error() -> None:
    async with _client(_json_route({"error": "boom"}, status_code=500)) as client:
        with pytest.raises(httpx.HTTPStatusError):
            await client.embed(["a"], "bge-m3")


async def test_embed_raises_on_missing_data_key() -> None:
    async with _client(_json_route({"unexpected": []})) as client:
        with pytest.raises(KeyError):
            await client.embed(["a"], "bge-m3")


# =============================================================================
# rerank
# =============================================================================


async def test_rerank_maps_scores_back_onto_document_order() -> None:
    payload = {
        "results": [
            {"index": 1, "relevance_score": 0.9},
            {"index": 0, "relevance_score": 0.1},
        ]
    }
    async with _client(_json_route(payload)) as client:
        scores = await client.rerank("q", ["first", "second"], "reranker")

    assert scores == [0.1, 0.9]


async def test_rerank_scores_default_to_zero_for_omitted_documents() -> None:
    """Scores are pre-filled with 0.0, so a router that scores only some
    documents yields a full-length list rather than a short one -
    hybrid.py zips this against its candidates with strict=True."""
    payload = {"results": [{"index": 1, "relevance_score": 0.7}]}
    async with _client(_json_route(payload)) as client:
        scores = await client.rerank("q", ["a", "b", "c"], "reranker")

    assert scores == [0.0, 0.7, 0.0]


async def test_rerank_raises_on_server_error() -> None:
    async with _client(_json_route({}, status_code=503)) as client:
        with pytest.raises(httpx.HTTPStatusError):
            await client.rerank("q", ["a"], "reranker")


# =============================================================================
# chat_json
# =============================================================================


def _chat_payload(content: str) -> dict[str, object]:
    return {"choices": [{"message": {"content": content}}]}


async def test_chat_json_parses_the_nested_content_document() -> None:
    payload = _chat_payload(json.dumps({"lex": "a", "vec": "b", "hyde": "c"}))
    async with _client(_json_route(payload)) as client:
        data = await client.chat_json([{"role": "user", "content": "hi"}], "gen", {"name": "s"})

    assert data == {"lex": "a", "vec": "b", "hyde": "c"}


async def test_chat_json_sends_the_schema_as_response_format() -> None:
    seen: dict[str, object] = {}
    schema = {"name": "expanded_query", "schema": {"type": "object"}}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, json=_chat_payload("{}"))

    async with _client(handler) as client:
        await client.chat_json([{"role": "user", "content": "hi"}], "gen", schema)

    assert seen["response_format"] == {"type": "json_schema", "json_schema": schema}
    assert seen["model"] == "gen"


async def test_chat_json_raises_value_error_on_malformed_content() -> None:
    """expand_query() catches ValueError to fall back to the unexpanded
    query - json.JSONDecodeError is a ValueError subclass, so a model that
    ignores the schema degrades instead of crashing the search."""
    async with _client(_json_route(_chat_payload("not json at all"))) as client:
        with pytest.raises(ValueError):
            await client.chat_json([{"role": "user", "content": "hi"}], "gen", {"name": "s"})


async def test_chat_json_raises_key_error_on_missing_choices() -> None:
    async with _client(_json_route({"choices": []})) as client:
        with pytest.raises(IndexError):
            await client.chat_json([{"role": "user", "content": "hi"}], "gen", {"name": "s"})


async def test_chat_json_raises_on_server_error() -> None:
    async with _client(_json_route({}, status_code=500)) as client:
        with pytest.raises(httpx.HTTPStatusError):
            await client.chat_json([{"role": "user", "content": "hi"}], "gen", {"name": "s"})


async def test_chat_json_propagates_timeout() -> None:
    """httpx.TimeoutException is an httpx.HTTPError, which expand_query()
    catches - a slow router degrades the search rather than failing it."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("too slow", request=request)

    async with _client(handler) as client:
        with pytest.raises(httpx.HTTPError):
            await client.chat_json([{"role": "user", "content": "hi"}], "gen", {"name": "s"})


# =============================================================================
# list_models / tokenize / lifecycle
# =============================================================================


async def test_list_models_returns_ids() -> None:
    payload = {"data": [{"id": "bge-m3-q8_0"}, {"id": "qwen2.5-3b-instruct-q4_k_m"}]}
    async with _client(_json_route(payload)) as client:
        assert await client.list_models() == ["bge-m3-q8_0", "qwen2.5-3b-instruct-q4_k_m"]


async def test_list_models_raises_when_router_is_down() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    async with _client(handler) as client:
        with pytest.raises(httpx.HTTPError):
            await client.list_models()


async def test_tokenize_returns_token_ids() -> None:
    async with _client(_json_route({"tokens": [1, 2, 3]})) as client:
        assert await client.tokenize("some text", "reranker") == [1, 2, 3]


async def test_tokenize_posts_content_and_model() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        assert request.url.path == "/tokenize"
        return httpx.Response(200, json={"tokens": []})

    async with _client(handler) as client:
        await client.tokenize("some text", "reranker")

    assert seen == {"model": "reranker", "content": "some text"}


async def test_base_url_trailing_slash_is_stripped() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "http://router.invalid/tokenize"
        return httpx.Response(200, json={"tokens": []})

    client = LlmClient("http://router.invalid/", transport=httpx.MockTransport(handler))
    async with client:
        await client.tokenize("x", "m")


async def test_aclose_closes_the_underlying_client() -> None:
    client = _client(_json_route({"tokens": []}))
    await client.aclose()

    with pytest.raises(RuntimeError):
        await client.tokenize("x", "m")
