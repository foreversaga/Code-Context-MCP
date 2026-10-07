#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

ENV_FILE="${CODE_CONTEXT_MPS_ENV_FILE:-.env.mps}"
if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

GPU_INDEX="${CODE_CONTEXT_GPU_INDEX:-0}"
HARD_LIMIT_MIB="${CODE_CONTEXT_MPS_HARD_LIMIT_MIB:-4096}"
COMPOSE=(docker compose --env-file "$ENV_FILE" -f compose.yml -f compose.mps.yml)

mpsctl() {
  sudo env \
    CUDA_MPS_PROTOCOL_VERSION=3 \
    CUDA_MPS_PIPE_DIRECTORY=/run/nvidia-mps \
    CUDA_MPS_LOG_DIRECTORY=/var/log/nvidia-mps \
    nvidia-cuda-mps-control "$@"
}

cid="$("${COMPOSE[@]}" ps -q code-context-mcp 2>/dev/null || true)"
if [[ -n "$cid" ]]; then
  pid="$(docker inspect -f '{{.State.Pid}}' "$cid" 2>/dev/null || true)"
  if [[ -n "$pid" && "$pid" != "0" && -r "/proc/$pid/cgroup" ]]; then
    cgroup_rel="$(awk -F: '$1 == "0" {print $3}' "/proc/$pid/cgroup")"
    cgroup_path="/sys/fs/cgroup$cgroup_rel"
    if [[ -d "$cgroup_path" ]]; then
      # Reset soft reservation before Docker deletes the dmem leaf so ancestor
      # reservations do not retain a stale contribution.
      sudo nvidia-smi memory-limits --set \
        --namespace "$cgroup_path" \
        --soft-limit 0 \
        --hard-limit "$HARD_LIMIT_MIB" \
        -i "$GPU_INDEX" >/dev/null 2>&1 || true
    fi
  fi
fi

"${COMPOSE[@]}" down

if command -v nvidia-cuda-mps-control >/dev/null 2>&1; then
  mpsctl server delete codecontext --force >/dev/null 2>&1 || true
fi

echo "Code Context MCP MPS v3 runtime stopped."
