import os

import pytest

from code_context_mcp import mps_entrypoint


def test_parse_cuda_version():
    assert mps_entrypoint._parse_cuda_version("13.4") == (13, 4)
    assert mps_entrypoint._parse_cuda_version("13.4.1") == (13, 4)
    assert mps_entrypoint._parse_cuda_version(None) == (0, 0)
    assert mps_entrypoint._parse_cuda_version("bad") == (0, 0)


def test_mps_gate_requires_start_script(monkeypatch):
    monkeypatch.delenv("CODE_CONTEXT_MPS_TOKEN", raising=False)

    with pytest.raises(SystemExit, match="start-mps-v3.sh"):
        mps_entrypoint._wait_for_mps_gate()


def test_mps_gate_requires_pipe_directory(monkeypatch):
    monkeypatch.setenv("CODE_CONTEXT_MPS_TOKEN", "test")
    monkeypatch.setenv("CUDA_MPS_PIPE_DIRECTORY", "")

    with pytest.raises(SystemExit, match="CUDA_MPS_PIPE_DIRECTORY"):
        mps_entrypoint._wait_for_mps_gate()
