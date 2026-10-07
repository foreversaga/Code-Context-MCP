from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    home: Path
    embedder: str = "embeddinggemma2"
    model_id: str = "google/embeddinggemma-2"
    dimensions: int = 256
    device: str | None = None
    host: str = "127.0.0.1"
    port: int = 7438

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            home=Path(os.getenv("CODE_CONTEXT_HOME", "~/.code-context-mcp")).expanduser(),
            embedder=os.getenv("CODE_CONTEXT_EMBEDDER", "embeddinggemma2"),
            model_id=os.getenv("CODE_CONTEXT_MODEL", "google/embeddinggemma-2"),
            dimensions=int(os.getenv("CODE_CONTEXT_DIMENSIONS", "256")),
            device=os.getenv("CODE_CONTEXT_DEVICE") or None,
            host=os.getenv("CODE_CONTEXT_HOST", "127.0.0.1"),
            port=int(os.getenv("CODE_CONTEXT_PORT", "7438")),
        )
