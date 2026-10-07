from __future__ import annotations

import hashlib
import heapq
import json
import os
import re
import sqlite3
import struct
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

EMBED_BATCH_CHUNKS = 8
MAX_FILE_BYTES = 2_000_000


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _pack_embedding(values: list[float]) -> bytes:
    return struct.pack(f"<{len(values)}f", *values)


def _dot_embedding(query: list[float], stored: object) -> float:
    if isinstance(stored, str):
        values = json.loads(stored)
        return sum(x * float(y) for x, y in zip(query, values, strict=False))
    if isinstance(stored, memoryview):
        stored = stored.tobytes()
    if isinstance(stored, bytearray):
        stored = bytes(stored)
    if isinstance(stored, bytes):
        values = memoryview(stored).cast("f")
        return sum(x * float(y) for x, y in zip(query, values, strict=False))
    raise TypeError(f"Unsupported embedding storage type: {type(stored)!r}")


class CodeContextService:
    def __init__(self, home: Path, embedder: Embedder, chunker: CodeChunker | None = None) -> None:
        self.home = home.expanduser().resolve()
        self.home.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder
        self.chunker = chunker or CodeChunker()
        self.lock = threading.RLock()
        self.db = sqlite3.connect(self.home / "index.sqlite3", check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self.db.executescript(
            """
            PRAGMA journal_mode=WAL;
            PRAGMA cache_size=-32768;
            PRAGMA temp_store=FILE;
            PRAGMA wal_autocheckpoint=1000;
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
                embedding BLOB NOT NULL
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
        for dirpath, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
            dirnames[:] = [name for name in dirnames if name not in DEFAULT_EXCLUDES]
            directory = Path(dirpath)
            for filename in filenames:
                path = directory / filename
                if not self.chunker.supports(path):
                    continue
                try:
                    if path.stat().st_size > MAX_FILE_BYTES:
                        continue
                    relative = path.relative_to(root).as_posix()
                except (OSError, ValueError):
                    continue
                yield path, relative

    def _delete_file_chunks(self, project_id: str, relative_path: str) -> None:
        self.db.execute(
            """
            DELETE FROM chunks_fts
            WHERE chunk_id IN (
                SELECT CAST(id AS TEXT)
                FROM chunks
                WHERE project_id = ? AND path = ?
            )
            """,
            (project_id, relative_path),
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

        for path, relative_path in self._iter_files(root):
            current.add(relative_path)
            try:
                raw = path.read_bytes()
            except OSError:
                continue
            if len(raw) > MAX_FILE_BYTES:
                continue

            digest = _sha256(raw)
            if not force and known.get(relative_path) == digest:
                continue

            content = raw.decode("utf-8", errors="ignore")
            chunks = self.chunker.chunk(relative_path, content)

            # Compute embeddings before touching the existing index. If model inference
            # fails, the previous index for this file remains usable.
            encoded: list[tuple[object, list[float]]] = []
            for start in range(0, len(chunks), EMBED_BATCH_CHUNKS):
                chunk_batch = chunks[start : start + EMBED_BATCH_CHUNKS]
                embeddings = self.embedder.embed_documents(
                    [
                        (f"{relative_path}::{chunk.symbol or chunk.kind}", chunk.content)
                        for chunk in chunk_batch
                    ]
                )
                encoded.extend(zip(chunk_batch, embeddings, strict=True))

            # Keep transactions file-scoped so large rebuilds do not grow one huge WAL.
            with self.db:
                self._delete_file_chunks(project_id, relative_path)
                for chunk, embedding in encoded:
                    cur = self.db.execute(
                        """
                        INSERT INTO chunks(
                            project_id, path, symbol, kind, start_line, end_line,
                            content, embedding
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
                            sqlite3.Binary(_pack_embedding(embedding)),
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

        missing_files = set(known) - current
        for missing in missing_files:
            with self.db:
                self._delete_file_chunks(project_id, missing)
                self.db.execute(
                    "DELETE FROM files WHERE project_id = ? AND path = ?",
                    (project_id, missing),
                )

        # Keep the WAL bounded after a large indexing pass.
        self.db.execute("PRAGMA wal_checkpoint(PASSIVE)")

        return {
            "indexed_files": indexed_files,
            "indexed_chunks": indexed_chunks,
            "removed_files": len(missing_files),
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
        if limit <= 0:
            return []

        query_vec = self.embedder.embed_query(query)
        top: list[tuple[float, int, str, str | None, str, int, int]] = []

        cursor = self.db.execute(
            """
            SELECT id, path, symbol, kind, start_line, end_line, embedding
            FROM chunks
            WHERE project_id = ?
            """,
            (project_id,),
        )
        for row in cursor:
            score = _dot_embedding(query_vec, row["embedding"])
            candidate = (
                score,
                int(row["id"]),
                row["path"],
                row["symbol"],
                row["kind"],
                int(row["start_line"]),
                int(row["end_line"]),
            )
            if len(top) < limit:
                heapq.heappush(top, candidate)
            elif candidate[:2] > top[0][:2]:
                heapq.heapreplace(top, candidate)

        ranked = sorted(top, reverse=True)
        ids = [item[1] for item in ranked]
        if not ids:
            return []

        placeholders = ",".join("?" for _ in ids)
        contents = {
            int(row["id"]): row["content"]
            for row in self.db.execute(
                f"SELECT id, content FROM chunks WHERE id IN ({placeholders})",
                ids,
            )
        }

        return [
            {
                "chunk_id": chunk_id,
                "path": path,
                "symbol": symbol,
                "kind": kind,
                "start_line": start_line,
                "end_line": end_line,
                "content": contents[chunk_id],
                "score": score,
            }
            for score, chunk_id, path, symbol, kind, start_line, end_line in ranked
        ]

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
            SELECT
                c.id, c.path, c.symbol, c.kind, c.start_line, c.end_line, c.content,
                bm25(chunks_fts) AS bm
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
            SELECT id, path, symbol, kind, start_line, end_line, content
            FROM chunks
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
            SELECT id, path, symbol, kind, start_line, end_line, content
            FROM chunks
            WHERE project_id = ? AND lower(content) LIKE lower(?)
            ORDER BY path, start_line
            LIMIT ?
            """,
            (project_id, f"%{symbol}%", limit),
        ).fetchall()
        return [self._row_result(row) for row in rows]

    def get_chunk(self, chunk_id: int) -> dict[str, Any]:
        row = self.db.execute(
            """
            SELECT id, path, symbol, kind, start_line, end_line, content
            FROM chunks
            WHERE id = ?
            """,
            (chunk_id,),
        ).fetchone()
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
