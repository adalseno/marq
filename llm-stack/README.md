# Optional local LLM stack

Entirely optional, for convenience - `qmd-py` has no local-model-loading
concept at all (see `llm/client.py`); it's just a plain HTTP client
against whatever `QMD_LLM_BASE_URL` points at. By default that's the real
`ubuserver.internal` router. This lets you run an equivalent router
locally instead (e.g. to work offline, or without depending on shared
infrastructure) - there's no data-safety motivation like there is for the
Postgres container, since the LLM router holds no state of ours.

## Setup

1. Download GGUF files matching `models/preset.ini`'s section names into
   `models/` (not checked into git - they're large binaries):
   - `bge-m3-q8_0.gguf` (embeddings)
   - `qwen3-reranker-0.6b-q8_0.gguf` (reranking)
   - `qwen2.5-3b-instruct-q4_k_m.gguf` (query expansion/chat)
2. Start it: `podman-compose --profile llm up -d llm`
3. Point `.env` at it: `QMD_LLM_BASE_URL=http://localhost:8099`

## GPU acceleration

The `server-vulkan` image (same one `ubuserver.internal` runs) works with
any Vulkan-capable GPU - NVIDIA, AMD, or Intel - unlike CUDA-only images.
`compose.yaml` passes through `/dev/dri` for this. If your dev machine has
no GPU at all, drop `--gpu-layers 99` from the `llm` service's command
(or switch the image to `ghcr.io/ggml-org/llama.cpp:server` for a CPU-only
build) - inference will just be slower.
