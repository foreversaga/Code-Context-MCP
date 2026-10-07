# Code Context MCP

Local-first, agent-agnostic code intelligence over MCP. One daemon and one shared index can be used by Claude Code, Codex, and Pi.

## MVP

- Streamable HTTP MCP at `http://127.0.0.1:7438/mcp`
- Multi-project registry
- Tree-sitter AST chunking for Python, JavaScript/TypeScript, Java, Go, Rust, C/C++, and C#
- Incremental file hashing and re-indexing
- EmbeddingGemma 2 text/code embeddings
- 256-d Matryoshka vectors by default
- SQLite persistence
- FAISS semantic vector search
- Hybrid FAISS + SQLite FTS5 retrieval
- Symbol lookup and reference search
- No Claude/Codex/Pi-specific logic inside the server

## Quick Start

Requirements: Python 3.12.

```bash
git clone https://github.com/foreversaga/Code-Context-MCP.git
cd Code-Context-MCP

python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[embedding]"

code-context-mcp
```

Default endpoint:

```text
http://127.0.0.1:7438/mcp
```

Default configuration:

```text
Data:       ~/.code-context-mcp
Model:      google/embeddinggemma-2
Mode:       text/code only
Dimensions: 256
Device:     CPU on Apple Silicon; auto elsewhere
```

The `embedding` extra includes Pillow and torchvision, which the EmbeddingGemma 2
processor requires even for text/code-only usage.

The model is loaded lazily and the first indexing operation may load or download it.

For complete setup, client configuration, indexing, multi-project usage, and troubleshooting, see [docs/USAGE.md](docs/USAGE.md).


## Docker on GB10

For GB10 with CUDA 13.4+, use the MPS v3 GPU runtime. It applies a hard device-memory limit to the container's real cgroup before EmbeddingGemma is allowed to start.

```bash
cp .env.mps.example .env.mps
# Edit CODE_CONTEXT_PROJECTS_ROOT in .env.mps.

bash scripts/start-mps-v3.sh
```

Defaults:

```text
GPU hard limit:      4096 MiB
GPU soft limit:      0
PyTorch fraction:    90% of the MPS-limited visible memory
Docker CPU memory:   8 GB
GPU backend:         NVIDIA MPS v3 memory partitioning
CUDA image:          nvcr.io/nvidia/pytorch:26.09-py3 (CUDA 13.4.1)
MCP:                 127.0.0.1:7438
```

The startup script creates an MPS v3 server/namespace, starts the container behind a gate, discovers its actual Docker cgroup, applies `nvidia-smi memory-limits --hard-limit`, verifies CUDA sees the capped memory, and only then releases the MCP process.

Stop with:

```bash
bash scripts/stop-mps-v3.sh
```

### CPU-only fallback

Use this when the host does not provide CUDA 13.4 MPS v3 memory partitioning:

```bash
cp .env.example .env
# Edit CODE_CONTEXT_PROJECTS_ROOT in .env.

docker compose up -d --build
docker compose ps
```

The default safety limits are:

```text
Memory limit:       8 GB
Memory + swap:      8 GB total
Memory reservation: 2 GB
CPU limit:          4 cores
PIDs:               256
Shared memory:      256 MB
Embedding device:   CPU
GPU access:          disabled
Root filesystem:    read-only
MCP bind address:    127.0.0.1:7438
```

Project source is mounted read-only at the same absolute path inside the container. Existing Claude Code, Codex, and Pi MCP configuration therefore remains unchanged.

Do **not** add `--gpus all` to the CPU-only fallback. Use `scripts/start-mps-v3.sh` for GPU acceleration so CUDA allocations are protected by an MPS v3 device-memory hard limit.

The Docker entrypoint fails closed when:

- no cgroup memory limit is detected
- `CODE_CONTEXT_DEVICE` is not `cpu`

An explicit `CODE_CONTEXT_ALLOW_UNBOUNDED=1` override exists for development, but disables these protections.

## MCP clients

### Claude Code

```bash
claude mcp add --transport http --scope user code-context http://127.0.0.1:7438/mcp
```

### Codex

Add to `~/.codex/config.toml`:

```toml
[mcp_servers.code-context]
url = "http://127.0.0.1:7438/mcp"
```

### Pi

Pi supports Streamable HTTP MCP natively. Add to `~/.pi/agent/mcp.json`:

```json
{
  "mcpServers": {
    "code-context": {
      "type": "http",
      "url": "http://127.0.0.1:7438/mcp"
    }
  }
}
```

## First project

After connecting the MCP server, ask the coding agent:

```text
Register the current repository as backend and index it with Code Context MCP.
```

The server will register the project and build its code index. Later indexing runs only process changed files unless `force=true` is used.

## Tools

- `register_project(path, project_id?)`
- `list_projects()`
- `remove_project(project_id)`
- `index_project(project_id, force=false)`
- `search_code(project_id, query, limit=10)`
- `search_text(project_id, query, limit=10)`
- `find_symbol(project_id, symbol, limit=20)`
- `find_references(project_id, symbol, limit=30)`
- `get_chunk(chunk_id)`
- `get_index_status(project_id)`

Recommended behavior:

1. Use `find_symbol` or `search_text` for exact identifiers.
2. Use `search_code` for concepts or behavior.
3. Use `find_references` before changing shared symbols.
4. Read the returned source before editing.

## Configuration

```bash
export CODE_CONTEXT_HOME=~/.code-context-mcp
export CODE_CONTEXT_MODEL=google/embeddinggemma-2
export CODE_CONTEXT_DIMENSIONS=256
# Optional override. Apple Silicon defaults to CPU for memory stability.
# export CODE_CONTEXT_DEVICE=mps
export CODE_CONTEXT_HOST=127.0.0.1
export CODE_CONTEXT_PORT=7438
```

For tests and smoke checks without loading EmbeddingGemma 2:

```bash
export CODE_CONTEXT_EMBEDDER=hash
```

## Architecture

```text
Claude Code --+
Codex --------+--> MCP Streamable HTTP --> Code Context MCP
Pi -----------+                              |
                                             +-- Tree-sitter AST
                                             +-- EmbeddingGemma 2
                                             +-- FAISS (one active project in RAM)
                                             +-- SQLite metadata + FTS5
                                             +-- symbol/reference index
                                             +-- incremental multi-project index
```

## CI

GitHub Actions runs lint and tests on Python 3.12. CI intentionally uses a deterministic fake embedder so the indexing/search/MCP workflow is tested without downloading model weights.


## Memory behavior

The semantic vector index uses FAISS instead of scanning every SQLite embedding in Python.

- embeddings are stored as compact float32 vectors
- only one project's FAISS index is kept resident at a time
- repository scans use a disk-backed SQLite TEMP table instead of Python path sets
- indexing is file-scoped and batched
- search responses are capped and return previews; use `get_chunk` for full content
- Apple Silicon defaults embedding inference to CPU to avoid long-running PyTorch MPS graph-cache growth

Existing SQLite embeddings can rebuild a missing or stale FAISS cache without re-running the embedding model.
