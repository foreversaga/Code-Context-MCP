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
- Hybrid semantic + SQLite FTS5 retrieval
- Symbol lookup and reference search
- No Claude/Codex/Pi-specific logic inside the server

## Install

```bash
git clone https://github.com/foreversaga/Code-Context-MCP.git
cd Code-Context-MCP
python -m venv .venv
source .venv/bin/activate
pip install -e ".[embedding]"
```

Start:

```bash
code-context-mcp
```

Defaults:

```text
MCP:        http://127.0.0.1:7438/mcp
Data:       ~/.code-context-mcp
Model:      google/embeddinggemma-2
Dimensions: 256
```

EmbeddingGemma 2 is loaded lazily. This project does not cast it to FP16.

## MCP clients

### Claude Code

```bash
claude mcp add --transport http --scope user code-context http://127.0.0.1:7438/mcp
```

### Codex

```toml
[mcp_servers.code-context]
url = "http://127.0.0.1:7438/mcp"
```

### Pi

Install an MCP client extension such as `pi-codemcp`, then point it at the same server:

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

## Tools

- `register_project(path, project_id?)`
- `list_projects()`
- `index_project(project_id, force=false)`
- `search_code(project_id, query, limit=10)`
- `search_text(project_id, query, limit=10)`
- `find_symbol(project_id, symbol, limit=20)`
- `find_references(project_id, symbol, limit=30)`
- `get_chunk(chunk_id)`
- `get_index_status(project_id)`

Recommended agent behavior:

1. Use `find_symbol` or `search_text` for exact identifiers.
2. Use `search_code` for concepts or behavior.
3. Read the returned chunk before editing source.
4. Use `find_references` when a change may affect callers.

## Configuration

```bash
export CODE_CONTEXT_HOME=~/.code-context-mcp
export CODE_CONTEXT_MODEL=google/embeddinggemma-2
export CODE_CONTEXT_DIMENSIONS=256
export CODE_CONTEXT_HOST=127.0.0.1
export CODE_CONTEXT_PORT=7438
```

For tests and smoke checks without downloading a model:

```bash
export CODE_CONTEXT_EMBEDDER=hash
```

## Architecture

```text
Claude Code ─┐
Codex ───────┼─ MCP Streamable HTTP ── Code Context MCP
Pi ──────────┘                              │
                                           ├─ Tree-sitter AST
                                           ├─ EmbeddingGemma 2
                                           ├─ SQLite FTS5
                                           ├─ Symbol/reference index
                                           └─ Incremental project index
```

## CI

GitHub Actions runs lint and tests on Python 3.11 and 3.12. CI intentionally uses a deterministic fake embedder so tests validate the indexing/search/MCP workflow without downloading model weights.
