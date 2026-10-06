from pathlib import Path

import pytest
from mcp import Client

from code_context_mcp.embedder import HashEmbedder
from code_context_mcp.index import CodeContextService
from code_context_mcp.server import configure_service, mcp


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_mcp_tools_are_callable(tmp_path: Path):
    project = tmp_path / "repo"
    project.mkdir()
    (project / "main.py").write_text("def hello():\n    return 'hello world'\n")

    service = CodeContextService(tmp_path / "home", HashEmbedder(32))
    configure_service(service)

    async with Client(mcp) as client:
        await client.call_tool("register_project", {"path": str(project), "project_id": "repo"})
        await client.call_tool("index_project", {"project_id": "repo"})
        result = await client.call_tool(
            "search_code", {"project_id": "repo", "query": "hello world", "limit": 5}
        )
        assert not result.is_error

    service.close()
