from __future__ import annotations

import os
from pathlib import Path

from .server import main


def _read_cgroup_memory_limit() -> int | None:
    candidates = (
        Path("/sys/fs/cgroup/memory.max"),
        Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
    )
    for path in candidates:
        try:
            raw = path.read_text().strip()
        except OSError:
            continue
        if raw == "max":
            return None
        try:
            value = int(raw)
        except ValueError:
            continue
        # cgroup v1 sometimes reports a huge sentinel for "unlimited".
        if value >= 1 << 60:
            return None
        return value
    return None


def _require_safe_runtime() -> None:
    if os.getenv("CODE_CONTEXT_ALLOW_UNBOUNDED") == "1":
        return

    device = os.getenv("CODE_CONTEXT_DEVICE", "cpu").lower()
    if device != "cpu":
        raise SystemExit(
            "Refusing to start safe Docker runtime with CODE_CONTEXT_DEVICE != cpu. "
            "GB10 CUDA allocations can bypass Docker memory cgroups. "
            "Set CODE_CONTEXT_ALLOW_UNBOUNDED=1 only if you accept that risk."
        )

    memory_limit = _read_cgroup_memory_limit()
    if memory_limit is None:
        raise SystemExit(
            "Refusing to start without a Docker/cgroup memory limit. "
            "Use docker compose or pass --memory and --memory-swap explicitly. "
            "Set CODE_CONTEXT_ALLOW_UNBOUNDED=1 only if you accept the risk."
        )


def docker_main() -> None:
    _require_safe_runtime()
    main()


if __name__ == "__main__":
    docker_main()
