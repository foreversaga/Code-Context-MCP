from __future__ import annotations

import hashlib
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
from .vector_index import FaissVectorStore


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
    ".gradle",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".nox",
    "__pycache__",
    "coverage",
    "out",
}

EMBED_BATCH_CHUNKS = 8
MAX_FILE_BYTES = 2_000_000
MAX_RESULTS = 50
MAX_SEARCH_POOL = 100
SEARCH_PREVIEW_CHARS = 2_000
MAX_QUERY_CHARS = 8_000
MAX_SYMBOL_CHARS = 512


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _pack_embedding(values: list[float]) -> bytes:
    return struct.pack(f"<{len(values)}f", *values)


def _bounded_limit(value: int, maximum: int = MAX_RESULTS) -> int:
    return max(0, min(int(value), maximum))


def _bounded_text(value: str, maximum: int) -> str:
    return value[:maximum]


def _unpack_embedding(stored: object) -> list[float]:
    if isinstance(stored, str):
        return [float(value) for value in json.loads(stored)]
    if isinstance(stored, memoryview):
        stored = stored.tobytes()
    if isinstance(stored, bytearray):
        stored = bytes(stored)
    if isinstance(stored, bytes):
        if len(stored) % 4 != 0:
            raise ValueError("Invalid float32 embedding blob length")
        count = len(stored) // 4
        return list(struct.unpack(f"<{count}f", stored))
    raise TypeError(f"Unsupported embedding storage type: {type(stored)!r}")


class CodeContextService:
    def __init__(self, home: Path, embedder: Embedder, chunker: CodeChunker | None = None) -> None:
        self.home = home.expanduser().resolve()
        self.home.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder
        self.chunker = chunker or CodeChunker()
        self.vectors = FaissVectorStore(self.home / "vectors", self.embedder.dimensions)
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
            CREATE TABLE IF NOT EXISTS vector_state (
                project_id TEXT PRIMARY KEY,
                revision INTEGER NOT NULL DEFAULT 0
            );
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
        self._cleanup_orphans()
        self.db.commit()

    def _cleanup_orphans(self) -> None:
        self.db.execute(
            """
            DELETE FROM chunks_fts
            WHERE chunk_id IN (
                SELECT CAST(c.id AS TEXT)
                FROM chunks c
                LEFT JOIN projects p ON p.project_id = c.project_id
                WHERE p.project_id IS NULL
            )
            """
        )
        self.db.execute(
            """
            DELETE FROM chunks
            WHERE project_id NOT IN (SELECT project_id FROM projects)
            """
        )
        self.db.execute(
            """
            DELETE FROM files
            WHERE project_id NOT IN (SELECT project_id FROM projects)
            """
        )
        self.db.execute(
            """
            DELETE FROM vector_state
            WHERE project_id NOT IN (SELECT project_id FROM projects)
            """
        )

    def close(self) -> None:
        self.vectors.close()
        self.db.close()

    def register_project(self, path: str, project_id: str | None = None) -> dict[str, str]:
        root = Path(path).expanduser().resolve()
        if not root.is_dir():
            raise ValueError(f"Project path does not exist or is not a directory: {root}")

        existing_by_path = self.db.execute(
            "SELECT project_id FROM projects WHERE root_path = ?",
            (str(root),),
        ).fetchone()
        if existing_by_path is not None:
            existing_id = existing_by_path["project_id"]
            if project_id is None or project_id == existing_id:
                return {"project_id": existing_id, "root_path": str(root)}
            raise ValueError(
                f"Project path is already registered as {existing_id!r}; "
                "remove it before registering a new project id"
            )

        if project_id is None:
            slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", root.name).strip("-").lower() or "project"
            project_id = f"{slug}-{_sha256(str(root).encode())[:8]}"

        existing_by_id = self.db.execute(
            "SELECT root_path FROM projects WHERE project_id = ?",
            (project_id,),
        ).fetchone()
        if existing_by_id is not None and existing_by_id["root_path"] != str(root):
            raise ValueError(
                f"Project id {project_id!r} is already registered for "
                f"{existing_by_id['root_path']!r}"
            )

        with self.db:
            self.db.execute(
                "INSERT INTO projects(project_id, root_path) VALUES (?, ?)",
                (project_id, str(root)),
            )
            self.db.execute(
                "INSERT OR IGNORE INTO vector_state(project_id, revision) VALUES (?, 0)",
                (project_id,),
            )
        return {"project_id": project_id, "root_path": str(root)}

    def remove_project(self, project_id: str) -> dict[str, int | str]:
        row = self.db.execute(
            "SELECT project_id FROM projects WHERE project_id = ?",
            (project_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"Unknown project: {project_id}")

        chunk_count = self.db.execute(
            "SELECT count(*) AS n FROM chunks WHERE project_id = ?",
            (project_id,),
        ).fetchone()["n"]
        file_count = self.db.execute(
            "SELECT count(*) AS n FROM files WHERE project_id = ?",
            (project_id,),
        ).fetchone()["n"]

        while True:
            rows = self.db.execute(
                "SELECT path FROM files WHERE project_id = ? ORDER BY path LIMIT 256",
                (project_id,),
            ).fetchall()
            if not rows:
                break
            for file_row in rows:
                relative_path = file_row["path"]
                with self.db:
                    self._delete_file_chunks(project_id, relative_path)
                    self.db.execute(
                        "DELETE FROM files WHERE project_id = ? AND path = ?",
                        (project_id, relative_path),
                    )

        # Clean any legacy chunks that are not represented in files metadata.
        with self.db:
            self.db.execute(
                """
                DELETE FROM chunks_fts
                WHERE chunk_id IN (
                    SELECT CAST(id AS TEXT)
                    FROM chunks
                    WHERE project_id = ?
                )
                """,
                (project_id,),
            )
            self.db.execute("DELETE FROM chunks WHERE project_id = ?", (project_id,))
            self.db.execute("DELETE FROM files WHERE project_id = ?", (project_id,))
            self.db.execute("DELETE FROM vector_state WHERE project_id = ?", (project_id,))
            self.db.execute("DELETE FROM projects WHERE project_id = ?", (project_id,))

        self.vectors.delete(project_id)
        self.db.execute("PRAGMA wal_checkpoint(PASSIVE)")
        return {
            "project_id": project_id,
            "removed_files": int(file_count),
            "removed_chunks": int(chunk_count),
        }

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

    def _vector_revision(self, project_id: str) -> int:
        row = self.db.execute(
            "SELECT revision FROM vector_state WHERE project_id = ?",
            (project_id,),
        ).fetchone()
        if row is None:
            with self.db:
                self.db.execute(
                    "INSERT INTO vector_state(project_id, revision) VALUES (?, 0)",
                    (project_id,),
                )
            return 0
        return int(row["revision"])

    def _bump_vector_revision(self, project_id: str) -> int:
        self.db.execute(
            """
            INSERT INTO vector_state(project_id, revision) VALUES (?, 1)
            ON CONFLICT(project_id) DO UPDATE SET revision = revision + 1
            """,
            (project_id,),
        )
        row = self.db.execute(
            "SELECT revision FROM vector_state WHERE project_id = ?",
            (project_id,),
        ).fetchone()
        return int(row["revision"])

    def _iter_project_embeddings(self, project_id: str):
        cursor = self.db.execute(
            "SELECT id, embedding FROM chunks WHERE project_id = ? ORDER BY id",
            (project_id,),
        )
        for row in cursor:
            yield int(row["id"]), _unpack_embedding(row["embedding"])

    def _ensure_vectors(self, project_id: str):
        revision = self._vector_revision(project_id)
        return self.vectors.ensure(
            project_id,
            revision,
            self._iter_project_embeddings(project_id),
        )

    def index_project(self, project_id: str, force: bool = False) -> dict[str, int]:
        root = self._project_root(project_id)
        indexed_files = 0
        indexed_chunks = 0
        removed_files = 0
        vector_changed = False

        # Track the current filesystem scan in SQLite instead of Python sets/dicts.
        # temp_store=FILE keeps very large repository inventories off the Python heap.
        self.db.execute(
            "CREATE TEMP TABLE IF NOT EXISTS current_scan(path TEXT PRIMARY KEY)"
        )
        self.db.execute("DELETE FROM current_scan")

        # FAISS is a cache over the persisted embeddings. If an older installation has
        # no index yet, this does a bounded streaming rebuild once.
        self._ensure_vectors(project_id)

        try:
            for path, relative_path in self._iter_files(root):
                self.db.execute(
                    "INSERT OR IGNORE INTO current_scan(path) VALUES (?)",
                    (relative_path,),
                )
                known = self.db.execute(
                    "SELECT sha256 FROM files WHERE project_id = ? AND path = ?",
                    (project_id, relative_path),
                ).fetchone()

                try:
                    raw = path.read_bytes()
                except OSError:
                    continue
                if len(raw) > MAX_FILE_BYTES:
                    continue

                digest = _sha256(raw)
                if not force and known is not None and known["sha256"] == digest:
                    continue

                content = raw.decode("utf-8", errors="ignore")
                chunks = self.chunker.chunk(relative_path, content)

                encoded: list[tuple[object, list[float]]] = []
                for batch_start in range(0, len(chunks), EMBED_BATCH_CHUNKS):
                    chunk_batch = chunks[batch_start : batch_start + EMBED_BATCH_CHUNKS]
                    embeddings = self.embedder.embed_documents(
                        [
                            (f"{relative_path}::{chunk.symbol or chunk.kind}", chunk.content)
                            for chunk in chunk_batch
                        ]
                    )
                    encoded.extend(zip(chunk_batch, embeddings, strict=True))

                old_ids = [
                    int(row["id"])
                    for row in self.db.execute(
                        "SELECT id FROM chunks WHERE project_id = ? AND path = ?",
                        (project_id, relative_path),
                    )
                ]
                new_ids: list[int] = []
                new_vectors: list[list[float]] = []

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
                        chunk_id = int(cur.lastrowid)
                        new_ids.append(chunk_id)
                        new_vectors.append(embedding)
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
                    revision = self._bump_vector_revision(project_id)

                self.vectors.remove_ids(project_id, old_ids)
                self.vectors.add(project_id, new_ids, new_vectors)
                self.vectors.set_revision(project_id, revision)
                vector_changed = True
                indexed_files += 1
                indexed_chunks += len(chunks)

            # Delete disappeared/now-excluded files in bounded batches instead of
            # materializing every missing path at once.
            while True:
                missing_rows = self.db.execute(
                    """
                    SELECT f.path
                    FROM files f
                    LEFT JOIN current_scan s ON s.path = f.path
                    WHERE f.project_id = ? AND s.path IS NULL
                    LIMIT 256
                    """,
                    (project_id,),
                ).fetchall()
                if not missing_rows:
                    break

                for row in missing_rows:
                    missing = row["path"]
                    old_ids = [
                        int(chunk_row["id"])
                        for chunk_row in self.db.execute(
                            "SELECT id FROM chunks WHERE project_id = ? AND path = ?",
                            (project_id, missing),
                        )
                    ]
                    with self.db:
                        self._delete_file_chunks(project_id, missing)
                        self.db.execute(
                            "DELETE FROM files WHERE project_id = ? AND path = ?",
                            (project_id, missing),
                        )
                        revision = self._bump_vector_revision(project_id)

                    self.vectors.remove_ids(project_id, old_ids)
                    self.vectors.set_revision(project_id, revision)
                    vector_changed = True
                    removed_files += 1

            if vector_changed:
                self.vectors.save(project_id, self._vector_revision(project_id))

            # current_scan writes use the same SQLite connection; finish that
            # transaction before asking WAL to checkpoint.
            self.db.commit()
            self.db.execute("PRAGMA wal_checkpoint(PASSIVE)")
            return {
                "indexed_files": indexed_files,
                "indexed_chunks": indexed_chunks,
                "removed_files": removed_files,
            }
        except Exception:
            # The SQLite revision is authoritative. Discard an in-memory FAISS index
            # that may have been only partially updated; the next access will rebuild
            # it from persisted embeddings if revisions differ.
            self.vectors.invalidate(project_id)
            raise
        finally:
            self.db.execute("DELETE FROM current_scan")
            self.db.commit()

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

        self._ensure_vectors(project_id)
        query_vec = self.embedder.embed_query(query)
        hits = self.vectors.search(project_id, query_vec, limit)
        ids = [chunk_id for chunk_id, _ in hits]
        if not ids:
            return []

        placeholders = ",".join("?" for _ in ids)
        rows = {
            int(row["id"]): row
            for row in self.db.execute(
                f"""
                SELECT id, path, symbol, kind, start_line, end_line,
                       substr(content, 1, ?) AS content
                FROM chunks
                WHERE id IN ({placeholders})
                """,
                [SEARCH_PREVIEW_CHARS, *ids],
            )
        }

        results = []
        for chunk_id, score in hits:
            row = rows.get(chunk_id)
            if row is not None:
                results.append(self._row_result(row, score))
        return results

    def _fts_query(self, query: str) -> str | None:
        tokens = re.findall(r"[A-Za-z0-9_]+", query)
        if not tokens:
            return None
        return " OR ".join(f'"{token.replace(chr(34), "")}"' for token in tokens[:16])

    def _lexical(self, project_id: str, query: str, limit: int) -> list[dict[str, Any]]:
        limit = _bounded_limit(limit, MAX_SEARCH_POOL)
        match = self._fts_query(query)
        if not match:
            return []
        rows = self.db.execute(
            """
            SELECT
                c.id, c.path, c.symbol, c.kind, c.start_line, c.end_line,
                substr(c.content, 1, 2000) AS content,
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
        query = _bounded_text(query, MAX_QUERY_CHARS)
        limit = _bounded_limit(limit)
        if limit == 0:
            return []
        pool = min(max(limit * 3, 20), MAX_SEARCH_POOL)
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
        query = _bounded_text(query, MAX_QUERY_CHARS)
        return self._lexical(project_id, query, _bounded_limit(limit))

    def find_symbol(self, project_id: str, symbol: str, limit: int = 20) -> list[dict[str, Any]]:
        symbol = _bounded_text(symbol, MAX_SYMBOL_CHARS)
        limit = _bounded_limit(limit)
        rows = self.db.execute(
            """
            SELECT id, path, symbol, kind, start_line, end_line,
                   substr(content, 1, 2000) AS content
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
        symbol = _bounded_text(symbol, MAX_SYMBOL_CHARS)
        limit = _bounded_limit(limit)
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", symbol):
            rows = self.db.execute(
                """
                SELECT c.id, c.path, c.symbol, c.kind, c.start_line, c.end_line,
                       substr(c.content, 1, 2000) AS content
                FROM chunks_fts
                JOIN chunks c ON c.id = CAST(chunks_fts.chunk_id AS INTEGER)
                WHERE chunks_fts.project_id = ? AND chunks_fts MATCH ?
                ORDER BY c.path, c.start_line
                LIMIT ?
                """,
                (project_id, f'"{symbol}"', limit),
            ).fetchall()
        else:
            rows = self.db.execute(
                """
                SELECT id, path, symbol, kind, start_line, end_line,
                       substr(content, 1, 2000) AS content
                FROM chunks
                WHERE project_id = ? AND content LIKE ?
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
            "vector_backend": "faiss",
            "vector_revision": self._vector_revision(project_id),
        }
