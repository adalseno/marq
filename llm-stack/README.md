# Optional local LLM stack

Entirely optional, for convenience - `marq` has no local-model-loading
concept at all (see `llm/client.py`); it's just a plain HTTP client
against whatever `MARQ_LLM_BASE_URL` points at, which defaults to
`http://localhost:8099` - a placeholder rather than a router that exists,
and exactly what this stack serves. So running this is one way to satisfy
the default; pointing at a shared router elsewhere is the other. There's
no data-safety motivation like there is for the Postgres container, since
the LLM router holds no state of ours.

## Setup

1. Download GGUF files matching `models/preset.ini`'s section names into
   `models/` (not checked into git - they're large binaries):
   - `bge-m3-q8_0.gguf` (embeddings)
   - `qwen3-reranker-0.6b-q8_0.gguf` (reranking)
   - `qwen2.5-3b-instruct-q4_k_m.gguf` (query expansion/chat)
2. Start it: `podman-compose --profile llm up -d llm`
3. Point `.env` at it: `MARQ_LLM_BASE_URL=http://localhost:8099`

## GPU acceleration

The `server-vulkan` image (same one the shared yourserver.com router
runs) works with
any Vulkan-capable GPU - NVIDIA, AMD, or Intel - unlike CUDA-only images.
`compose.yaml` passes through `/dev/dri` for this. If your dev machine has
no GPU at all, drop `--gpu-layers 99` from the `llm` service's command
(or switch the image to `ghcr.io/ggml-org/llama.cpp:server` for a CPU-only
build) - inference will just be slower.
