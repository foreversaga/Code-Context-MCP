# Usage Guide

Code Context MCP runs as one local MCP daemon and shares the same code index across Claude Code, Codex, and Pi.

## 1. Install

Requirements:

- Python 3.12
- macOS or Linux
- Git

```bash
git clone https://github.com/foreversaga/Code-Context-MCP.git
cd Code-Context-MCP

python3.12 -m venv .venv
source .venv/bin/activate

pip install -e ".[embedding]"
```

The `embedding` extra installs the model processor dependencies, including Pillow
and torchvision. EmbeddingGemma 2 requires these even for text/code-only usage.

## 2. Start the server

Run the daemon in a dedicated terminal:

```bash
source .venv/bin/activate
code-context-mcp
```

Default endpoint:

```text
http://127.0.0.1:7438/mcp
```

Default data directory:

```text
~/.code-context-mcp
```

Default embedding configuration:

```text
Model:      google/embeddinggemma-2
Mode:       text/code only
Dimensions: 256
Device:     CPU on Apple Silicon; auto elsewhere
```

The model is loaded lazily. The first project indexing operation may take longer because the model must be loaded or downloaded.

On Apple Silicon, embedding inference defaults to CPU. PyTorch MPS can retain memory for varying input shapes during long-running inference workloads. You can explicitly opt in to MPS with `CODE_CONTEXT_DEVICE=mps` if you want to test it on your installed PyTorch version.



## 2A. Docker deployment on GB10

For a DGX Spark / GB10 host, Docker is recommended because it can place a hard cgroup limit around all CPU-side allocations from Python, FAISS, SQLite, Torch CPU inference, and filesystem cache charged to the container.

The safe configuration deliberately does **not** expose the NVIDIA GPU. This matters on GB10: CPU and GPU share unified physical memory, but CUDA allocations are not reliably constrained by Docker's memory cgroup. A container using CUDA can therefore consume unified memory beyond `--memory`.

### Start with Docker Compose

Create the local environment file:

```bash
cp .env.example .env
```

Edit `.env` and set the absolute directory containing the repositories you want to index:

```text
CODE_CONTEXT_PROJECTS_ROOT=/home/barry/projects
```

Do not use `~` in this value. Use an absolute path.

Start:

```bash
docker compose up -d --build
```

Check status:

```bash
docker compose ps
docker compose logs -f code-context-mcp
```

Stop:

```bash
docker compose down
```

The index and Hugging Face model cache are kept in named Docker volumes, so `docker compose down` does not delete them.

### Default resource guardrails

```text
CODE_CONTEXT_MEMORY_LIMIT=8g
CODE_CONTEXT_MEMORY_RESERVATION=2g
CODE_CONTEXT_CPU_LIMIT=4.0
CODE_CONTEXT_CPU_THREADS=4
CODE_CONTEXT_PORT=7438
```

Compose applies:

```text
memory limit       8 GB
memory + swap      8 GB
CPU                4 cores
pids               256
shm                256 MB
device             CPU
GPU                not exposed
root filesystem    read-only
project source     read-only
network bind       localhost only
```

Setting `memswap_limit` equal to `mem_limit` prevents the container from pushing additional anonymous memory into swap when it reaches the limit.

### Fail-closed startup

The Docker image checks its runtime before starting the MCP server. Unless `CODE_CONTEXT_ALLOW_UNBOUNDED=1` is explicitly set, it refuses to run when:

1. no cgroup memory limit is detected
2. `CODE_CONTEXT_DEVICE` is anything other than `cpu`

This prevents an accidental plain `docker run` without `--memory`, and prevents an accidental switch to CUDA where the GB10 unified-memory allocation would bypass the cgroup limit.

### Direct docker run

Compose is preferred, but the equivalent safe shape is:

```bash
docker build -t code-context-mcp:local .

docker run --rm \
  --name code-context-mcp \
  --memory=8g \
  --memory-swap=8g \
  --cpus=4 \
  --pids-limit=256 \
  --shm-size=256m \
  --read-only \
  --tmpfs /tmp:rw,size=256m \
  -p 127.0.0.1:7438:7438 \
  -v code-context-data:/data \
  -v code-context-hf:/cache/huggingface \
  -v /home/barry/projects:/home/barry/projects:ro \
  code-context-mcp:local
```

Replace `/home/barry/projects` with your actual absolute projects root.

### Register projects

Because the host project root is mounted to the same absolute path inside the container, continue using normal absolute paths:

```text
register_project(path="/home/barry/projects/backend", project_id="backend")
index_project(project_id="backend")
```

Claude Code, Codex, and Pi still connect to:

```text
http://127.0.0.1:7438/mcp
```

### GB10 warning

Do not add `--gpus all` to this safe deployment. On GB10 unified memory, Docker's `--memory` limit does not reliably cap CUDA-side allocations. If GPU embedding is intentionally enabled, host OOM protection must be handled separately and the Docker safety guarantee no longer applies.

## 3. Connect an MCP client

All clients connect to the same local endpoint.

### Claude Code

Add the server for the current user:

```bash
claude mcp add --transport http --scope user code-context http://127.0.0.1:7438/mcp
```

Verify it from Claude Code with:

```text
/mcp
```

### Codex

Add this to `~/.codex/config.toml`:

```toml
[mcp_servers.code-context]
url = "http://127.0.0.1:7438/mcp"
```

Then verify:

```bash
codex mcp list
```

You can also inspect MCP tools from the Codex TUI with:

```text
/mcp
```

### Pi

Pi supports Streamable HTTP MCP natively.

User-level config:

```text
~/.pi/agent/mcp.json
```

Project-level config:

```text
.pi/mcp.json
```

Example:

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

Project-level MCP configuration is loaded after the project is trusted.

## 4. Register and index a repository

After connecting the MCP server, you normally do this through the coding agent.

Example prompt:

```text
Register the current repository as backend and index it with Code Context MCP.
```

The agent should call:

```text
register_project(path="<absolute repo path>", project_id="backend")
index_project(project_id="backend")
```

You only need to register a repository once.

After files change, refresh the index with:

```text
Update the Code Context MCP index for backend.
```

The server hashes files and only re-indexes changed files.

To force a full rebuild:

```text
Force rebuild the Code Context MCP index for backend.
```

This maps to:

```text
index_project(project_id="backend", force=true)
```

## 5. Daily usage

Use normal natural-language coding requests. The agent can decide which MCP search tool to call.

### Concept or behavior search

Example:

```text
Where is JWT token validation implemented?
```

Preferred tool:

```text
search_code
```

### Exact symbol search

Example:

```text
Find AuthService and validateJwtToken.
```

Preferred tools:

```text
find_symbol
search_text
```

### Find callers or impact

Example:

```text
Find every place that references validateJwtToken before changing it.
```

Preferred tool:

```text
find_references
```

### Read the exact indexed code

Search results contain a `chunk_id`. The agent can retrieve the complete indexed chunk with:

```text
get_chunk
```

## 6. Recommended agent workflow

For code changes, the intended workflow is:

```text
Question / task
    |
    +-- exact name? ----> find_symbol / search_text
    |
    +-- concept? -------> search_code
    |
    +-- changing API? --> find_references
                            |
                            v
                         get_chunk
                            |
                            v
                      read source file
                            |
                            v
                          edit
```

Code Context MCP is a retrieval layer. The coding agent should still read the actual source file before editing it.

## 7. Multiple projects

Run one MCP daemon for all repositories.

Example project IDs:

```text
backend
frontend
payment-service
game-server
```

Register each repository once:

```text
Register /Users/me/projects/backend as backend.
Register /Users/me/projects/frontend as frontend.
Register /Users/me/projects/payment-service as payment-service.
```

All three MCP clients reuse the same project registry and SQLite index:

```text
Claude Code --+
Codex --------+--> Code Context MCP --> backend
Pi -----------+                     --> frontend
                                    --> payment-service
                                    --> game-server
```

List registered projects with:

```text
list_projects
```

Remove an unused project and its persisted chunks, embeddings, and FTS data with:

```text
remove_project(project_id="backend")
```

## 8. Check index status

Ask the agent:

```text
Show the Code Context MCP index status for backend.
```

This calls:

```text
get_index_status(project_id="backend")
```

The result includes:

- project path
- indexed file count
- chunk count
- embedding dimensions

## 9. Configuration

Environment variables:

```bash
export CODE_CONTEXT_HOME=~/.code-context-mcp
export CODE_CONTEXT_MODEL=google/embeddinggemma-2
export CODE_CONTEXT_DIMENSIONS=256
# Optional. Apple Silicon defaults to CPU for long-running memory stability.
# export CODE_CONTEXT_DEVICE=mps
export CODE_CONTEXT_HOST=127.0.0.1
export CODE_CONTEXT_PORT=7438
```

For development or smoke tests without loading EmbeddingGemma 2:

```bash
export CODE_CONTEXT_EMBEDDER=hash
code-context-mcp
```

The hash embedder is only for testing. Do not use it for real semantic code search.

## 10. Available MCP tools

| Tool | Purpose |
| --- | --- |
| `register_project` | Register a repository path |
| `list_projects` | List registered repositories |
| `remove_project` | Remove a project and delete its persisted index |
| `index_project` | Incrementally index or force rebuild a project |
| `search_code` | Hybrid semantic + lexical code search |
| `search_text` | Exact / lexical full-text search |
| `find_symbol` | Find indexed symbols |
| `find_references` | Find chunks referencing a symbol |
| `get_chunk` | Read a full indexed chunk |
| `get_index_status` | Show project index statistics |

## 11. Troubleshooting

### MCP client cannot connect

Confirm the server is running and listening on port 7438:

```bash
lsof -i :7438
```

Then confirm the client is configured with:

```text
http://127.0.0.1:7438/mcp
```

### Project not found

List registered projects:

```text
list_projects
```

If needed, register it again with the intended `project_id`.

### Search results look stale

Run:

```text
index_project(project_id="<project>")
```

For a full rebuild:

```text
index_project(project_id="<project>", force=true)
```

### Port 7438 is already in use

Choose another port before starting the server:

```bash
export CODE_CONTEXT_PORT=7440
code-context-mcp
```

Then update all MCP clients to:

```text
http://127.0.0.1:7440/mcp
```

### Embedding model is unavailable

Confirm the embedding dependencies were installed:

```bash
pip install -e ".[embedding]"
```

Then restart the MCP server.

## 12. Architecture

```text
Claude Code --+
Codex --------+--> Streamable HTTP --> Code Context MCP
Pi -----------+                         |
                                        +-- Tree-sitter AST chunks
                                        +-- EmbeddingGemma 2
                                        +-- FAISS semantic index
                                        +-- SQLite metadata + FTS5
                                        +-- symbol/reference index
                                        +-- incremental multi-project index
```


## 13. Memory behavior

Code Context MCP keeps semantic retrieval bounded for large repositories:

- FAISS performs vector top-k search instead of loading/scanning all embeddings in Python.
- Only one project's FAISS index is resident in the daemon at a time.
- Embeddings are persisted as compact float32 BLOBs in SQLite so a FAISS cache can be rebuilt without re-embedding source code.
- FAISS cache files include a SQLite vector revision. If indexing is interrupted and revisions differ, the cache is rebuilt by streaming persisted embeddings.
- Repository file discovery uses a disk-backed SQLite TEMP table instead of retaining all paths/hashes in Python sets.
- Files over 2 MB and generated dependency/build directories are excluded from indexing.
- Embedding inference uses small batches and a 2048-token maximum sequence length.
- Search results are capped at 50 results and 2,000-character previews. Use `get_chunk` when full indexed content is needed.

At 256 dimensions, an exact float32 FAISS index uses roughly 1 KB per indexed chunk, excluding small FAISS metadata overhead. Switching between projects unloads the previous in-memory index.
