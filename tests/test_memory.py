import json
from pathlib import Path

from code_context_mcp.chunker import CodeChunker
from code_context_mcp.embedder import EmbeddingGemma2Embedder, HashEmbedder
from code_context_mcp.index import CodeContextService


def test_embeddings_are_stored_as_float32_blobs(tmp_path: Path):
    project = tmp_path / "repo"
    project.mkdir()
    (project / "main.py").write_text("def hello():\n    return 'hello world'\n")

    service = CodeContextService(tmp_path / "home", HashEmbedder(64))
    service.register_project(str(project), "repo")
    service.index_project("repo")

    row = service.db.execute(
        "SELECT typeof(embedding) AS storage_type, length(embedding) AS bytes FROM chunks LIMIT 1"
    ).fetchone()

    assert row["storage_type"] == "blob"
    assert row["bytes"] == 64 * 4
    service.close()


def test_legacy_json_embeddings_remain_searchable(tmp_path: Path):
    project = tmp_path / "repo"
    project.mkdir()
    (project / "main.py").write_text("def authenticate():\n    return 'jwt validation'\n")

    embedder = HashEmbedder(32)
    service = CodeContextService(tmp_path / "home", embedder)
    service.register_project(str(project), "repo")
    service.index_project("repo")

    row = service.db.execute("SELECT id, embedding FROM chunks LIMIT 1").fetchone()
    raw = memoryview(row["embedding"]).cast("f")
    legacy = json.dumps([float(v) for v in raw])
    with service.db:
        service.db.execute(
            "UPDATE chunks SET embedding = ? WHERE id = ?",
            (legacy, row["id"]),
        )

    results = service.search_code("repo", "jwt validation", limit=5)
    assert results
    assert results[0]["path"] == "main.py"
    service.close()


def test_large_semantic_chunks_are_bounded():
    body = "\n".join(f"        value += {i}" for i in range(5000))
    source = (
        "class LargeService:\n"
        "    def calculate(self):\n"
        "        value = 0\n"
        f"{body}\n"
        "        return value\n"
    )

    chunker = CodeChunker(max_chunk_chars=4000)
    chunks = chunker.chunk("large.py", source)

    assert chunks
    assert max(len(chunk.content) for chunk in chunks) <= 4000
    assert any(chunk.kind.endswith("_part") for chunk in chunks)


class _FakeModel:
    def __init__(self):
        self.calls = []

    def encode(self, payload, **kwargs):
        self.calls.append((payload, kwargs))
        return [[1.0, 0.0, 0.0, 0.0] for _ in payload]


def test_embeddinggemma_uses_small_batches():
    embedder = EmbeddingGemma2Embedder(dimensions=128, batch_size=4)
    fake = _FakeModel()
    embedder._model = fake

    embedder.embed_documents([("a.py", "def a(): pass"), ("b.py", "def b(): pass")])

    assert fake.calls[0][1]["batch_size"] == 4
