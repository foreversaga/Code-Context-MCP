from __future__ import annotations

from mcp.server import MCPServer

from .config import Settings
from .embedder import build_embedder
from .index import CodeContextService


mcp = MCPServer(
    "Code Context MCP",
    instructions=(
        "Local code intelligence for coding agents. Prefer find_symbol/search_text for exact "
        "identifiers and search_code for conceptual queries. Read returned chunks before editing."
    ),
)

_service: CodeContextService | None = None


def get_service() -> CodeContextService:
    global _service
    if _service is None:
        settings = Settings.from_env()
        _service = CodeContextService(
            settings.home,
            build_embedder(settings.embedder, settings.model_id, settings.dimensions),
        )
    return _service


def configure_service(service: CodeContextService) -> None:
    global _service
    _service = service


def _call(method: str, *args, **kwargs):
    service = get_service()
    with service.lock:
        return getattr(service, method)(*args, **kwargs)


@mcp.tool()
def register_project(path: str, project_id: str | None = None):
    """Register a local repository path and return its stable project id."""
    return _call("register_project", path, project_id)


@mcp.tool()
def list_projects():
    """List registered code projects."""
    return _call("list_projects")


@mcp.tool()
def remove_project(project_id: str):
    """Remove a project and delete all of its persisted chunks, embeddings, and FTS data."""
    return _call("remove_project", project_id)


@mcp.tool()
def index_project(project_id: str, force: bool = False):
    """Incrementally index a registered project. Set force=true for a full rebuild."""
    return _call("index_project", project_id, force)


@mcp.tool()
def search_code(project_id: str, query: str, limit: int = 10):
    """Hybrid semantic + lexical code search for concepts, behavior, or implementation locations."""
    return _call("search_code", project_id, query, limit)


@mcp.tool()
def search_text(project_id: str, query: str, limit: int = 10):
    """Lexical full-text search, best for exact identifiers, error messages, and config values."""
    return _call("search_text", project_id, query, limit)


@mcp.tool()
def find_symbol(project_id: str, symbol: str, limit: int = 20):
    """Find function, method, class, interface, or other indexed symbols."""
    return _call("find_symbol", project_id, symbol, limit)


@mcp.tool()
def find_references(project_id: str, symbol: str, limit: int = 30):
    """Find indexed code chunks that reference a symbol."""
    return _call("find_references", project_id, symbol, limit)


@mcp.tool()
def get_chunk(chunk_id: int):
    """Read one indexed code chunk by id."""
    return _call("get_chunk", chunk_id)


@mcp.tool()
def get_index_status(project_id: str):
    """Return index statistics for one project."""
    return _call("get_index_status", project_id)


def main() -> None:
    settings = Settings.from_env()
    mcp.run(
        transport="streamable-http",
        host=settings.host,
        port=settings.port,
    )


if __name__ == "__main__":
    main()
