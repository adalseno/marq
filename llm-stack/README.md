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
   (Docker: `docker compose --profile llm up -d llm` — see the host
   requirements below, which bind harder here than for Postgres.)
3. Point `.env` at it: `MARQ_LLM_BASE_URL=http://localhost:8099`

## Host requirements (podman or Docker, but Linux either way)

The Postgres container in `compose.yaml` is portable to Docker with no
change beyond the command name. This service is not, because two of its
settings are Linux-host concepts rather than runtime concepts:

- `devices: - /dev/dri` passes through the GPU. There is no `/dev/dri`
  inside Docker Desktop's VM on macOS or Windows, so the service won't
  start there under either runtime.
- The `:Z` on the `./llm-stack/models` mount asks for SELinux
  relabelling. Docker honours `:Z` on an SELinux host and ignores it
  elsewhere, so it is harmless off Fedora/RHEL — just inert.

None of this is a reason to avoid Docker for marq itself: this stack is
optional, nothing depends on it, and `MARQ_LLM_BASE_URL` can point at any
OpenAI-shaped endpoint. On a machine without a usable GPU — including
Docker Desktop — drop `--gpu-layers 99` or use the CPU-only image, as
described below, and it runs anywhere.

## GPU acceleration

The `server-vulkan` image (same one the shared yourserver.com router
runs) works with
any Vulkan-capable GPU - NVIDIA, AMD, or Intel - unlike CUDA-only images.
`compose.yaml` passes through `/dev/dri` for this. If your dev machine has
no GPU at all, drop `--gpu-layers 99` from the `llm` service's command
(or switch the image to `ghcr.io/ggml-org/llama.cpp:server` for a CPU-only
build) - inference will just be slower.
