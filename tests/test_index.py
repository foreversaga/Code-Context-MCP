from pathlib import Path

from code_context_mcp.embedder import HashEmbedder
from code_context_mcp.index import CodeContextService


JAVA = """
package demo;

public class AuthService {
    public boolean validateJwtToken(String token) {
        String message = "validate JWT token";
        return token != null && !token.isBlank();
    }
}

class LoginHandler {
    private final AuthService authService = new AuthService();

    public boolean login(String token) {
        return authService.validateJwtToken(token);
    }
}
"""

TS = """
export function issueReward(userId: string): string {
  const message = "issue reward after settled order";
  return userId + message;
}
"""


def make_project(root: Path) -> None:
    (root / "src").mkdir()
    (root / "src" / "AuthService.java").write_text(JAVA)
    (root / "src" / "reward.ts").write_text(TS)


def test_index_search_and_incremental_update(tmp_path: Path):
    project = tmp_path / "project"
    project.mkdir()
    make_project(project)

    service = CodeContextService(tmp_path / "home", HashEmbedder(64))
    registered = service.register_project(str(project), "demo")

    assert registered["project_id"] == "demo"

    first = service.index_project("demo")
    assert first["indexed_files"] == 2
    assert first["indexed_chunks"] >= 3

    second = service.index_project("demo")
    assert second["indexed_files"] == 0

    semantic = service.search_code("demo", "JWT token validation", limit=5)
    assert semantic
    assert semantic[0]["path"].endswith("AuthService.java")

    symbols = service.find_symbol("demo", "validateJwtToken")
    assert any(item["symbol"] == "validateJwtToken" for item in symbols)

    refs = service.find_references("demo", "validateJwtToken")
    assert len(refs) >= 2

    status = service.get_index_status("demo")
    assert status["files"] == 2
    assert status["chunks"] >= 3

    reward = project / "src" / "reward.ts"
    reward.write_text(TS + "\nexport const rewardVersion = 2;\n")
    changed = service.index_project("demo")
    assert changed["indexed_files"] == 1

    service.close()
