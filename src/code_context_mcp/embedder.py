from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from typing import Protocol


class Embedder(Protocol):
    dimensions: int

    def embed_query(self, text: str) -> list[float]: ...

    def embed_documents(self, items: Sequence[tuple[str, str]]) -> list[list[float]]: ...


def _normalize(values: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(float(v) * float(v) for v in values)) or 1.0
    return [float(v) / norm for v in values]


class HashEmbedder:
    """Deterministic lightweight embedder for tests and offline smoke checks."""

    def __init__(self, dimensions: int = 64) -> None:
        self.dimensions = dimensions

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dimensions
        tokens = re.findall(r"[A-Za-z0-9_]+", text.lower())
        for token in tokens:
            digest = hashlib.sha256(token.encode()).digest()
            idx = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = -1.0 if digest[4] & 1 else 1.0
            vec[idx] += sign
        return _normalize(vec)

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def embed_documents(self, items: Sequence[tuple[str, str]]) -> list[list[float]]:
        return [self._embed(f"{title} {content}") for title, content in items]


class EmbeddingGemma2Embedder:
    """Lazy SentenceTransformers adapter for Google's EmbeddingGemma 2 text/code backbone."""

    def __init__(
        self,
        model_id: str = "google/embeddinggemma-2",
        dimensions: int = 256,
        batch_size: int = 2,
        max_seq_length: int = 2048,
    ) -> None:
        if dimensions not in (128, 256, 512, 768):
            raise ValueError("EmbeddingGemma 2 dimensions must be 128, 256, 512, or 768")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        self.model_id = model_id
        self.dimensions = dimensions
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    'EmbeddingGemma 2 requires: pip install "code-context-mcp[embedding]"'
                ) from exc
            self._model = SentenceTransformer(
                self.model_id,
                config_kwargs={"vision_config": None, "audio_config": None},
                truncate_dim=self.dimensions,
            )
            self._model.max_seq_length = min(
                int(getattr(self._model, "max_seq_length", self.max_seq_length)),
                self.max_seq_length,
            )
        return self._model

    def _convert(self, row) -> list[float]:
        values = row.tolist() if hasattr(row, "tolist") else list(row)
        return _normalize(values[: self.dimensions])

    def embed_query(self, text: str) -> list[float]:
        model = self._load()
        rows = model.encode(
            [text],
            prompt_name="CodeRetrieval",
            normalize_embeddings=True,
            convert_to_numpy=True,
            batch_size=1,
            truncate_dim=self.dimensions,
        )
        return self._convert(rows[0])

    def embed_documents(self, items: Sequence[tuple[str, str]]) -> list[list[float]]:
        model = self._load()
        payload = [f"title: {title or 'none'} | text: {content}" for title, content in items]
        rows = model.encode(
            payload,
            normalize_embeddings=True,
            convert_to_numpy=True,
            batch_size=self.batch_size,
            truncate_dim=self.dimensions,
        )
        return [self._convert(row) for row in rows]


def build_embedder(name: str, model_id: str, dimensions: int) -> Embedder:
    if name == "hash":
        return HashEmbedder(dimensions=min(dimensions, 256))
    if name == "embeddinggemma2":
        return EmbeddingGemma2Embedder(model_id=model_id, dimensions=dimensions)
    raise ValueError(f"Unknown embedder: {name}")
