from __future__ import annotations

import json
import os
from collections.abc import Iterable
from pathlib import Path

import faiss
import numpy as np


class FaissVectorStore:
    """Disk-backed FAISS cache with at most one project index resident in memory."""

    def __init__(self, root: Path, dimensions: int, rebuild_batch: int = 512) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.dimensions = dimensions
        self.rebuild_batch = rebuild_batch
        self._project_id: str | None = None
        self._index = None
        self._revision: int | None = None
        self._dirty = False

    def _key(self, project_id: str) -> str:
        import hashlib

        return hashlib.sha256(project_id.encode()).hexdigest()[:24]

    def _paths(self, project_id: str) -> tuple[Path, Path]:
        key = self._key(project_id)
        return self.root / f"{key}.faiss", self.root / f"{key}.json"

    def _new_index(self):
        return faiss.IndexIDMap2(faiss.IndexFlatIP(self.dimensions))

    def _read_revision(self, project_id: str) -> int | None:
        _, meta_path = self._paths(project_id)
        try:
            data = json.loads(meta_path.read_text())
            if data.get("project_id") != project_id:
                return None
            if int(data.get("dimensions", -1)) != self.dimensions:
                return None
            return int(data["revision"])
        except (OSError, ValueError, TypeError, json.JSONDecodeError, KeyError):
            return None

    def _drop_active(self) -> None:
        self._project_id = None
        self._index = None
        self._revision = None
        self._dirty = False

    def _flush_active_if_dirty(self) -> None:
        if self._dirty and self._project_id is not None and self._revision is not None:
            self.save(self._project_id, self._revision)

    def ensure(
        self,
        project_id: str,
        revision: int,
        source: Iterable[tuple[int, list[float]]],
    ):
        if self._project_id == project_id and self._revision == revision and self._index is not None:
            return self._index

        self._flush_active_if_dirty()
        self._drop_active()

        index_path, _ = self._paths(project_id)
        disk_revision = self._read_revision(project_id)
        if index_path.exists() and disk_revision == revision:
            try:
                index = faiss.read_index(str(index_path))
                if index.d != self.dimensions:
                    raise ValueError("FAISS dimension mismatch")
                self._project_id = project_id
                self._index = index
                self._revision = revision
                return index
            except Exception:
                # Treat a corrupt/incompatible index as a rebuildable cache miss.
                pass

        index = self._new_index()
        ids: list[int] = []
        vectors: list[list[float]] = []

        def flush_batch() -> None:
            if not ids:
                return
            matrix = np.asarray(vectors, dtype=np.float32)
            id_array = np.asarray(ids, dtype=np.int64)
            index.add_with_ids(matrix, id_array)
            ids.clear()
            vectors.clear()

        for chunk_id, embedding in source:
            ids.append(int(chunk_id))
            vectors.append(embedding)
            if len(ids) >= self.rebuild_batch:
                flush_batch()
        flush_batch()

        self._project_id = project_id
        self._index = index
        self._revision = revision
        self._dirty = True
        self.save(project_id, revision)
        return index

    def remove_ids(self, project_id: str, ids: list[int]) -> None:
        if not ids:
            return
        if self._project_id != project_id or self._index is None:
            raise RuntimeError("FAISS project is not active")
        id_array = np.asarray(ids, dtype=np.int64)
        self._index.remove_ids(id_array)
        self._dirty = True

    def add(self, project_id: str, ids: list[int], vectors: list[list[float]]) -> None:
        if not ids:
            return
        if self._project_id != project_id or self._index is None:
            raise RuntimeError("FAISS project is not active")
        matrix = np.asarray(vectors, dtype=np.float32)
        id_array = np.asarray(ids, dtype=np.int64)
        self._index.add_with_ids(matrix, id_array)
        self._dirty = True

    def set_revision(self, project_id: str, revision: int) -> None:
        if self._project_id != project_id or self._index is None:
            raise RuntimeError("FAISS project is not active")
        self._revision = revision
        self._dirty = True

    def search(self, project_id: str, query: list[float], limit: int) -> list[tuple[int, float]]:
        if self._project_id != project_id or self._index is None:
            raise RuntimeError("FAISS project is not active")
        if limit <= 0 or self._index.ntotal == 0:
            return []

        query_matrix = np.asarray([query], dtype=np.float32)
        scores, ids = self._index.search(query_matrix, min(limit, self._index.ntotal))
        return [
            (int(chunk_id), float(score))
            for chunk_id, score in zip(ids[0], scores[0], strict=True)
            if int(chunk_id) >= 0
        ]

    def save(self, project_id: str, revision: int) -> None:
        if self._project_id != project_id or self._index is None:
            return

        index_path, meta_path = self._paths(project_id)
        tmp_index = index_path.with_suffix(".faiss.tmp")
        tmp_meta = meta_path.with_suffix(".json.tmp")

        faiss.write_index(self._index, str(tmp_index))
        tmp_meta.write_text(
            json.dumps(
                {
                    "project_id": project_id,
                    "revision": int(revision),
                    "dimensions": self.dimensions,
                }
            )
        )
        os.replace(tmp_index, index_path)
        os.replace(tmp_meta, meta_path)
        self._revision = revision
        self._dirty = False

    def invalidate(self, project_id: str) -> None:
        if self._project_id == project_id:
            self._drop_active()

    def delete(self, project_id: str) -> None:
        if self._project_id == project_id:
            self._drop_active()
        for path in self._paths(project_id):
            try:
                path.unlink()
            except FileNotFoundError:
                pass

    def close(self) -> None:
        self._flush_active_if_dirty()
        self._drop_active()
