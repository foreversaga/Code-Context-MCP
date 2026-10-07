from __future__ import annotations

import os
import time
from pathlib import Path


def _parse_cuda_version(value: str | None) -> tuple[int, int]:
    if not value:
        return (0, 0)
    pieces = value.split(".")
    try:
        return (int(pieces[0]), int(pieces[1]))
    except (IndexError, ValueError):
        return (0, 0)


def _wait_for_mps_gate() -> None:
    token = os.getenv("CODE_CONTEXT_MPS_TOKEN")
    if not token:
        raise SystemExit(
            "CODE_CONTEXT_MPS_TOKEN is missing. Start the GPU runtime with "
            "scripts/start-mps-v3.sh so the device-memory hard limit is applied first."
        )

    pipe_value = os.getenv("CUDA_MPS_PIPE_DIRECTORY", "")
    if not pipe_value:
        raise SystemExit("CUDA_MPS_PIPE_DIRECTORY is required for the MPS v3 runtime.")
    pipe_dir = Path(pipe_value)

    control_socket = pipe_dir / "control"
    deadline = time.monotonic() + int(os.getenv("CODE_CONTEXT_MPS_GATE_TIMEOUT", "120"))
    ready_file = Path("/data") / f".mps-v3-ready-{token}"

    while time.monotonic() < deadline:
        if ready_file.exists() and control_socket.exists():
            return
        time.sleep(0.25)

    raise SystemExit(
        "Timed out waiting for the host to apply and verify the MPS v3 GPU memory hard limit."
    )


def _verify_cuda_runtime() -> None:
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available inside the MPS v3 container.")

    cuda_version = _parse_cuda_version(torch.version.cuda)
    if cuda_version < (13, 4):
        raise SystemExit(
            f"CUDA {torch.version.cuda!r} is too old. MPS v3 memory partitioning "
            "requires CUDA 13.4 or newer."
        )

    fraction = float(os.getenv("CODE_CONTEXT_TORCH_GPU_FRACTION", "0.90"))
    if not 0.0 < fraction <= 1.0:
        raise SystemExit("CODE_CONTEXT_TORCH_GPU_FRACTION must be > 0 and <= 1.")

    # Secondary guardrail inside the MPS v3 cgroup hard limit. The MPS hard limit
    # remains authoritative; this keeps PyTorch below that ceiling during normal use.
    torch.cuda.memory.set_per_process_memory_fraction(fraction)


def main() -> None:
    _wait_for_mps_gate()
    _verify_cuda_runtime()

    from .server import main as server_main

    server_main()


if __name__ == "__main__":
    main()
