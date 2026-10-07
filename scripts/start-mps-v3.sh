#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

ENV_FILE="${CODE_CONTEXT_MPS_ENV_FILE:-.env.mps}"
if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing $ENV_FILE. Copy .env.mps.example to .env.mps and edit it first." >&2
  exit 1
fi

set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

: "${CODE_CONTEXT_PROJECTS_ROOT:?Set CODE_CONTEXT_PROJECTS_ROOT in $ENV_FILE}"

GPU_INDEX="${CODE_CONTEXT_GPU_INDEX:-0}"
HARD_LIMIT_MIB="${CODE_CONTEXT_MPS_HARD_LIMIT_MIB:-4096}"
SOFT_LIMIT_MIB="${CODE_CONTEXT_MPS_SOFT_LIMIT_MIB:-0}"
export CODE_CONTEXT_MPS_TOKEN="$(date +%s)-$$"
export CODE_CONTEXT_MPS_PIPE_DIRECTORY="/run/nvidia-mps/codecontext/mcp"

COMPOSE=(docker compose --env-file "$ENV_FILE" -f compose.yml -f compose.mps.yml)

for command in docker sudo nvidia-smi nvidia-cuda-mps-control; do
  command -v "$command" >/dev/null 2>&1 || {
    echo "Missing required command: $command" >&2
    exit 1
  }
done

if ! mountpoint -q /sys/fs/cgroup || [[ "$(stat -fc %T /sys/fs/cgroup)" != "cgroup2fs" ]]; then
  echo "MPS v3 memory partitioning requires cgroup v2 at /sys/fs/cgroup." >&2
  exit 1
fi

if ! nvidia-smi memory-limits --help >/dev/null 2>&1; then
  echo "This NVIDIA driver does not expose 'nvidia-smi memory-limits'." >&2
  echo "Update the DGX Spark driver/CUDA stack to a CUDA 13.4-compatible release." >&2
  exit 1
fi

existing_cid="$("${COMPOSE[@]}" ps -q code-context-mcp 2>/dev/null || true)"
if [[ -n "$existing_cid" ]] && [[ "$(docker inspect -f '{{.State.Running}}' "$existing_cid")" == "true" ]]; then
  echo "code-context-mcp is already running. Stop it with scripts/stop-mps-v3.sh first." >&2
  exit 1
fi

mpsctl() {
  sudo env \
    CUDA_MPS_PROTOCOL_VERSION=3 \
    CUDA_MPS_PIPE_DIRECTORY=/run/nvidia-mps \
    CUDA_MPS_LOG_DIRECTORY=/var/log/nvidia-mps \
    nvidia-cuda-mps-control "$@"
}

sudo mkdir -p /run/nvidia-mps /var/log/nvidia-mps

if ! mpsctl server list >/dev/null 2>&1; then
  if pgrep -f '[n]vidia-cuda-mps-control' >/dev/null 2>&1; then
    echo "An incompatible/legacy MPS daemon is already running." >&2
    echo "Stop it before starting the MPS v3 runtime." >&2
    exit 1
  fi
  sudo env \
    CUDA_MPS_PIPE_DIRECTORY=/run/nvidia-mps \
    CUDA_MPS_LOG_DIRECTORY=/var/log/nvidia-mps \
    nvidia-cuda-mps-control -d -p 3
fi

if mpsctl server list codecontext >/dev/null 2>&1; then
  mpsctl server delete codecontext --force
fi

mpsctl server create codecontext --uid=10001 --allowed-devices="$GPU_INDEX"
mpsctl namespace create mcp --server=codecontext
mpsctl namespace set mcp --server=codecontext --pinned-memory-limit="${HARD_LIMIT_MIB}M"

pipe_dir="$(mpsctl namespace get mcp codecontext pipe-directory | tail -n 1 | tr -d '[:space:]')"
if [[ "$pipe_dir" != "$CODE_CONTEXT_MPS_PIPE_DIRECTORY" ]]; then
  echo "Unexpected MPS namespace pipe directory: $pipe_dir" >&2
  mpsctl server delete codecontext --force || true
  exit 1
fi

cleanup_on_error() {
  status=$?
  if [[ $status -eq 0 ]]; then
    return
  fi
  echo "MPS v3 startup failed; cleaning up container/server." >&2
  cid="$("${COMPOSE[@]}" ps -q code-context-mcp 2>/dev/null || true)"
  if [[ -n "$cid" ]]; then
    pid="$(docker inspect -f '{{.State.Pid}}' "$cid" 2>/dev/null || true)"
    if [[ -n "$pid" && "$pid" != "0" && -r "/proc/$pid/cgroup" ]]; then
      rel="$(awk -F: '$1 == "0" {print $3}' "/proc/$pid/cgroup")"
      cgroup="/sys/fs/cgroup$rel"
      sudo nvidia-smi memory-limits --set \
        --namespace "$cgroup" --soft-limit 0 --hard-limit "$HARD_LIMIT_MIB" -i "$GPU_INDEX" \
        >/dev/null 2>&1 || true
    fi
  fi
  "${COMPOSE[@]}" down >/dev/null 2>&1 || true
  mpsctl server delete codecontext --force >/dev/null 2>&1 || true
  exit "$status"
}
trap cleanup_on_error ERR INT TERM

"${COMPOSE[@]}" up -d --build code-context-mcp

cid="$("${COMPOSE[@]}" ps -q code-context-mcp)"
if [[ -z "$cid" ]]; then
  echo "Failed to obtain container id." >&2
  exit 1
fi

pid="$(docker inspect -f '{{.State.Pid}}' "$cid")"
if [[ -z "$pid" || "$pid" == "0" ]]; then
  echo "Container is not running." >&2
  exit 1
fi

cgroup_rel="$(awk -F: '$1 == "0" {print $3}' "/proc/$pid/cgroup")"
cgroup_path="/sys/fs/cgroup$cgroup_rel"
if [[ ! -d "$cgroup_path" ]]; then
  echo "Cannot locate container cgroup: $cgroup_path" >&2
  exit 1
fi

sudo nvidia-smi memory-limits --set \
  --namespace "$cgroup_path" \
  --soft-limit "$SOFT_LIMIT_MIB" \
  --hard-limit "$HARD_LIMIT_MIB" \
  -i "$GPU_INDEX"

limits="$(sudo nvidia-smi memory-limits --get --namespace "$cgroup_path" -i "$GPU_INDEX")"
printf '%s\n' "$limits"
if ! grep -qi 'hard limit:.*MiB' <<<"$limits"; then
  echo "Unable to verify an active GPU hard limit." >&2
  exit 1
fi
if grep -qi 'hard limit:.*max' <<<"$limits"; then
  echo "GPU hard limit is still unlimited." >&2
  exit 1
fi

visible_total="$(docker exec "$cid" python -c 'import torch; print(torch.cuda.mem_get_info()[1])')"
max_expected="$((HARD_LIMIT_MIB * 1024 * 1024 + 128 * 1024 * 1024))"
if (( visible_total > max_expected )); then
  echo "CUDA still reports ${visible_total} bytes, larger than the configured hard limit." >&2
  exit 1
fi

docker exec "$cid" sh -c "touch /data/.mps-v3-ready-$CODE_CONTEXT_MPS_TOKEN"

for _ in $(seq 1 60); do
  health="$(docker inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}' "$cid")"
  if [[ "$health" == "healthy" ]]; then
    trap - ERR INT TERM
    echo
    echo "Code Context MCP is healthy with MPS v3 GPU memory partitioning."
    echo "MCP: http://127.0.0.1:${CODE_CONTEXT_PORT:-7438}/mcp"
    echo "GPU hard limit: ${HARD_LIMIT_MIB} MiB"
    echo "GPU cgroup: $cgroup_path"
    exit 0
  fi
  if [[ "$health" == "unhealthy" ]]; then
    docker logs "$cid"
    exit 1
  fi
  sleep 1
done

docker logs "$cid"
echo "Container did not become healthy in time." >&2
exit 1
