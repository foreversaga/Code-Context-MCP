import pytest

from code_context_mcp import docker_entrypoint


def test_safe_docker_runtime_accepts_cpu_with_memory_limit(monkeypatch):
    monkeypatch.setenv("CODE_CONTEXT_DEVICE", "cpu")
    monkeypatch.delenv("CODE_CONTEXT_ALLOW_UNBOUNDED", raising=False)
    monkeypatch.setattr(
        docker_entrypoint,
        "_read_cgroup_memory_limit",
        lambda: 8 * 1024**3,
    )

    docker_entrypoint._require_safe_runtime()


def test_safe_docker_runtime_rejects_cuda(monkeypatch):
    monkeypatch.setenv("CODE_CONTEXT_DEVICE", "cuda")
    monkeypatch.delenv("CODE_CONTEXT_ALLOW_UNBOUNDED", raising=False)
    monkeypatch.setattr(
        docker_entrypoint,
        "_read_cgroup_memory_limit",
        lambda: 8 * 1024**3,
    )

    with pytest.raises(SystemExit, match="CODE_CONTEXT_DEVICE != cpu"):
        docker_entrypoint._require_safe_runtime()


def test_safe_docker_runtime_rejects_unlimited_memory(monkeypatch):
    monkeypatch.setenv("CODE_CONTEXT_DEVICE", "cpu")
    monkeypatch.delenv("CODE_CONTEXT_ALLOW_UNBOUNDED", raising=False)
    monkeypatch.setattr(docker_entrypoint, "_read_cgroup_memory_limit", lambda: None)

    with pytest.raises(SystemExit, match="memory limit"):
        docker_entrypoint._require_safe_runtime()


def test_unsafe_override_is_explicit(monkeypatch):
    monkeypatch.setenv("CODE_CONTEXT_ALLOW_UNBOUNDED", "1")
    monkeypatch.setenv("CODE_CONTEXT_DEVICE", "cuda")
    monkeypatch.setattr(docker_entrypoint, "_read_cgroup_memory_limit", lambda: None)

    docker_entrypoint._require_safe_runtime()
