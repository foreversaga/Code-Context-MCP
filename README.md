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
```

The model is loaded lazily and the first indexing operation may load or download it.

For complete setup, client configuration, indexing, multi-project usage, and troubleshooting, see [docs/USAGE.md](docs/USAGE.md).

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
                                             +-- SQLite FTS5
                                             +-- symbol/reference index
                                             +-- incremental multi-project index
```

## CI

GitHub Actions runs lint and tests on Python 3.12. CI intentionally uses a deterministic fake embedder so the indexing/search/MCP workflow is tested without downloading model weights.
