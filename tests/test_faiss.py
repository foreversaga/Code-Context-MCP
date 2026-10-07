from pathlib import Path

from code_context_mcp.embedder import HashEmbedder
from code_context_mcp.index import CodeContextService


def _make_repo(path: Path, body: str) -> None:
    path.mkdir()
    (path / "main.py").write_text(body)


def test_faiss_cache_is_created_and_reused(tmp_path: Path):
    project = tmp_path / "repo"
    _make_repo(project, "def authenticate():\n    return 'jwt validation'\n")
    home = tmp_path / "home"

    service = CodeContextService(home, HashEmbedder(32))
    service.register_project(str(project), "repo")
    service.index_project("repo")
    first = service.search_code("repo", "jwt validation", limit=5)

    assert first
    assert first[0]["path"] == "main.py"
    assert list((home / "vectors").glob("*.faiss"))
    service.close()

    reopened = CodeContextService(home, HashEmbedder(32))
    second = reopened.search_code("repo", "jwt validation", limit=5)

    assert second
    assert second[0]["path"] == "main.py"
    assert reopened.get_index_status("repo")["vector_backend"] == "faiss"
    reopened.close()


def test_incremental_index_removes_old_faiss_ids(tmp_path: Path):
    project = tmp_path / "repo"
    _make_repo(project, "def old_symbol():\n    return 'legacy auth path'\n")

    service = CodeContextService(tmp_path / "home", HashEmbedder(64))
    service.register_project(str(project), "repo")
    service.index_project("repo")

    old_results = service.search_code("repo", "legacy auth path", limit=5)
    old_ids = {item["chunk_id"] for item in old_results}
    assert old_ids

    (project / "main.py").write_text("def new_symbol():\n    return 'modern payment flow'\n")
    changed = service.index_project("repo")

    assert changed["indexed_files"] == 1
    new_results = service.search_code("repo", "modern payment flow", limit=5)
    new_ids = {item["chunk_id"] for item in new_results}
    assert new_ids
    assert old_ids.isdisjoint(new_ids)
    assert service.find_symbol("repo", "old_symbol") == []
    service.close()


def test_missing_faiss_cache_rebuilds_from_sqlite_embeddings(tmp_path: Path):
    project = tmp_path / "repo"
    _make_repo(project, "def reward():\n    return 'settled order reward'\n")
    home = tmp_path / "home"

    service = CodeContextService(home, HashEmbedder(32))
    service.register_project(str(project), "repo")
    service.index_project("repo")
    service.close()

    for path in (home / "vectors").glob("*"):
        path.unlink()

    reopened = CodeContextService(home, HashEmbedder(32))
    results = reopened.search_code("repo", "settled order reward", limit=5)

    assert results
    assert results[0]["path"] == "main.py"
    assert list((home / "vectors").glob("*.faiss"))
    reopened.close()


def test_only_one_project_faiss_index_is_resident(tmp_path: Path):
    repo_a = tmp_path / "a"
    repo_b = tmp_path / "b"
    _make_repo(repo_a, "def alpha():\n    return 'alpha service'\n")
    _make_repo(repo_b, "def beta():\n    return 'beta service'\n")

    service = CodeContextService(tmp_path / "home", HashEmbedder(32))
    service.register_project(str(repo_a), "a")
    service.register_project(str(repo_b), "b")
    service.index_project("a")
    service.index_project("b")

    service.search_code("a", "alpha service", limit=5)
    assert service.vectors._project_id == "a"

    service.search_code("b", "beta service", limit=5)
    assert service.vectors._project_id == "b"
    service.close()
