from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
from pathlib import Path
from typing import Any

from .chunker import CodeChunker
from .embedder import Embedder


DEFAULT_EXCLUDES = {
    ".git",
    ".idea",
    ".vscode",
    ".venv",
    "venv",
    "node_modules",
    "dist",
    "build",
    "target",
    "vendor",
    ".next",
    ".cache",
    "coverage",
}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=False))


class CodeContextService:
    def __init__(self, home: Path, embedder: Embedder, chunker: CodeChunker | None = None) -> None:
        self.home = home.expanduser().resolve()
        self.home.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder
        self.chunker = chunker or CodeChunker()
        self.lock = threading.RLock()\n        self.db = sqlite3.connect(self.home / "index.sqlite3", check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self.db.executescript(
            """
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS projects (
                project_id TEXT PRIMARY KEY,
                root_path TEXT NOT NULL UNIQUE
            );
            CREATE TABLE IF NOT EXISTS files (
                project_id TEXT NOT NULL,
                path TEXT NOT NULL,
                sha256 TEXT NOT NULL,
                PRIMARY KEY(project_id, path)
            );
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id TEXT NOT NULL,
                path TEXT NOT NULL,
                symbol TEXT,
                kind TEXT NOT NULL,
                start_line INTEGER NOT NULL,
                end_line INTEGER NOT NULL,
                content TEXT NOT NULL,
                embedding TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_chunks_project ON chunks(project_id);
            CREATE INDEX IF NOT EXISTS idx_chunks_symbol ON chunks(project_id, symbol);
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                chunk_id UNINDEXED,
                project_id UNINDEXED,
                path,
                symbol,
                content,
                tokenize='unicode61'
            );
            """
        )
        self.db.commit()

    def close(self) -> None:
        self.db.close()

    def register_project(self, path: str, project_id: str | None = None) -> dict[str, str]:
        root = Path(path).expanduser().resolve()
        if not root.is_dir():
            raise ValueError(f"Project path does not exist or is not a directory: {root}")
        if project_id is None:
            slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", root.name).strip("-").lower() or "project"
            project_id = f"{slug}-{_sha256(str(root).encode())[:8]}"
        with self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO projects(project_id, root_path) VALUES (?, ?)",
                (project_id, str(root)),
            )
        return {"project_id": project_id, "root_path": str(root)}

    def list_projects(self) -> list[dict[str, str]]:
        rows = self.db.execute(
            "SELECT project_id, root_path FROM projects ORDER BY project_id"
        ).fetchall()
        return [dict(row) for row in rows]

    def _project_root(self, project_id: str) -> Path:
        row = self.db.execute(
            "SELECT root_path FROM projects WHERE project_id = ?", (project_id,)
        ).fetchone()
        if row is None:
            raise ValueError(f"Unknown project: {project_id}")
        return Path(row["root_path"])

    def _iter_files(self, root: Path):
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            try:
                relative = path.relative_to(root)
            except ValueError:
                continue
            if any(part in DEFAULT_EXCLUDES for part in relative.parts):
                continue
            if self.chunker.supports(path):
                yield path, relative.as_posix()

    def _delete_file_chunks(self, project_id: str, relative_path: str) -> None:
        ids = [
            row["id"]
            for row in self.db.execute(
                "SELECT id FROM chunks WHERE project_id = ? AND path = ?",
                (project_id, relative_path),
            )
        ]
        if ids:
            placeholders = ",".join("?" for _ in ids)
            self.db.execute(
                f"DELETE FROM chunks_fts WHERE chunk_id IN ({placeholders})",
                [str(i) for i in ids],
            )
        self.db.execute(
            "DELETE FROM chunks WHERE project_id = ? AND path = ?",
            (project_id, relative_path),
        )

    def index_project(self, project_id: str, force: bool = False) -> dict[str, int]:
        root = self._project_root(project_id)
        known = {
            row["path"]: row["sha256"]
            for row in self.db.execute(
                "SELECT path, sha256 FROM files WHERE project_id = ?", (project_id,)
            )
        }
        current: set[str] = set()
        indexed_files = 0
        indexed_chunks = 0

        with self.db:
            for path, relative_path in self._iter_files(root):
                current.add(relative_path)
                raw = path.read_bytes()
                digest = _sha256(raw)
                if not force and known.get(relative_path) == digest:
                    continue

                content = raw.decode("utf-8", errors="ignore")
                chunks = self.chunker.chunk(relative_path, content)
                embeddings = self.embedder.embed_documents(
                    [
                        (f"{relative_path}::{chunk.symbol or chunk.kind}", chunk.content)
                        for chunk in chunks
                    ]
                )

                self._delete_file_chunks(project_id, relative_path)
                for chunk, embedding in zip(chunks, embeddings, strict=True):
                    cur = self.db.execute(
                        """
                        INSERT INTO chunks(
                            project_id, path, symbol, kind, start_line, end_line, content, embedding
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            project_id,
                            chunk.path,
                            chunk.symbol,
                            chunk.kind,
                            chunk.start_line,
                            chunk.end_line,
                            chunk.content,
                            json.dumps(embedding),
                        ),
                    )
                    chunk_id = cur.lastrowid
                    self.db.execute(
                        """
                        INSERT INTO chunks_fts(chunk_id, project_id, path, symbol, content)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            str(chunk_id),
                            project_id,
                            chunk.path,
                            chunk.symbol or "",
                            chunk.content,
                        ),
                    )
                self.db.execute(
                    """
                    INSERT INTO files(project_id, path, sha256) VALUES (?, ?, ?)
                    ON CONFLICT(project_id, path) DO UPDATE SET sha256 = excluded.sha256
                    """,
                    (project_id, relative_path, digest),
                )
                indexed_files += 1
                indexed_chunks += len(chunks)

            for missing in set(known) - current:
                self._delete_file_chunks(project_id, missing)
                self.db.execute(
                    "DELETE FROM files WHERE project_id = ? AND path = ?",
                    (project_id, missing),
                )

        return {
            "indexed_files": indexed_files,
            "indexed_chunks": indexed_chunks,
            "removed_files": len(set(known) - current),
        }

    def _row_result(self, row: sqlite3.Row, score: float = 0.0) -> dict[str, Any]:
        return {
            "chunk_id": row["id"],
            "path": row["path"],
            "symbol": row["symbol"],
            "kind": row["kind"],
            "start_line": row["start_line"],
            "end_line": row["end_line"],
            "content": row["content"],
            "score": score,
        }

    def _semantic(self, project_id: str, query: str, limit: int) -> list[dict[str, Any]]:
        query_vec = self.embedder.embed_query(query)
        rows = self.db.execute(
            "SELECT * FROM chunks WHERE project_id = ?", (project_id,)
        ).fetchall()
        scored = [
            (row, _cosine(query_vec, json.loads(row["embedding"])))
            for row in rows
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        return [self._row_result(row, score) for row, score in scored[:limit]]

    def _fts_query(self, query: str) -> str | None:
        tokens = re.findall(r"[A-Za-z0-9_]+", query)
        if not tokens:
            return None
        return " OR ".join(f'"{token.replace(chr(34), "")}"' for token in tokens[:16])

    def _lexical(self, project_id: str, query: str, limit: int) -> list[dict[str, Any]]:
        match = self._fts_query(query)
        if not match:
            return []
        rows = self.db.execute(
            """
            SELECT c.*, bm25(chunks_fts) AS bm
            FROM chunks_fts
            JOIN chunks c ON c.id = CAST(chunks_fts.chunk_id AS INTEGER)
            WHERE chunks_fts.project_id = ? AND chunks_fts MATCH ?
            ORDER BY bm ASC
            LIMIT ?
            """,
            (project_id, match, limit),
        ).fetchall()
        return [self._row_result(row, -float(row["bm"])) for row in rows]

    def search_code(self, project_id: str, query: str, limit: int = 10) -> list[dict[str, Any]]:
        pool = max(limit * 3, 20)
        semantic = self._semantic(project_id, query, pool)
        lexical = self._lexical(project_id, query, pool)
        combined: dict[int, dict[str, Any]] = {}
        scores: dict[int, float] = {}

        for rank, result in enumerate(semantic, start=1):
            cid = int(result["chunk_id"])
            combined[cid] = result
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (60 + rank)
        for rank, result in enumerate(lexical, start=1):
            cid = int(result["chunk_id"])
            combined[cid] = result
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (60 + rank)

        ranked = sorted(combined.values(), key=lambda r: scores[int(r["chunk_id"])], reverse=True)
        for result in ranked:
            result["score"] = scores[int(result["chunk_id"])]
        return ranked[:limit]

    def search_text(self, project_id: str, query: str, limit: int = 10) -> list[dict[str, Any]]:
        return self._lexical(project_id, query, limit)

    def find_symbol(self, project_id: str, symbol: str, limit: int = 20) -> list[dict[str, Any]]:
        rows = self.db.execute(
            """
            SELECT * FROM chunks
            WHERE project_id = ? AND symbol IS NOT NULL
              AND lower(symbol) LIKE lower(?)
            ORDER BY CASE WHEN lower(symbol) = lower(?) THEN 0 ELSE 1 END, path, start_line
            LIMIT ?
            """,
            (project_id, f"%{symbol}%", symbol, limit),
        ).fetchall()
        return [self._row_result(row) for row in rows]

    def find_references(self, project_id: str, symbol: str, limit: int = 30) -> list[dict[str, Any]]:
        rows = self.db.execute(
            """
            SELECT * FROM chunks
            WHERE project_id = ? AND lower(content) LIKE lower(?)
            ORDER BY path, start_line
            LIMIT ?
            """,
            (project_id, f"%{symbol}%", limit),
        ).fetchall()
        return [self._row_result(row) for row in rows]

    def get_chunk(self, chunk_id: int) -> dict[str, Any]:
        row = self.db.execute("SELECT * FROM chunks WHERE id = ?", (chunk_id,)).fetchone()
        if row is None:
            raise ValueError(f"Unknown chunk: {chunk_id}")
        return self._row_result(row)

    def get_index_status(self, project_id: str) -> dict[str, Any]:
        root = self._project_root(project_id)
        files = self.db.execute(
            "SELECT count(*) AS n FROM files WHERE project_id = ?", (project_id,)
        ).fetchone()["n"]
        chunks = self.db.execute(
            "SELECT count(*) AS n FROM chunks WHERE project_id = ?", (project_id,)
        ).fetchone()["n"]
        return {
            "project_id": project_id,
            "root_path": str(root),
            "files": files,
            "chunks": chunks,
            "embedding_dimensions": self.embedder.dimensions,
        }
