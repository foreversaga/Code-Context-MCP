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
```

The model is loaded lazily. The first project indexing operation may take longer because the model must be loaded or downloaded.

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
                                        +-- SQLite FTS5
                                        +-- symbol/reference index
                                        +-- incremental multi-project index
```
