FROM nvcr.io/nvidia/pytorch:26.09-py3

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CODE_CONTEXT_HOME=/data \
    CODE_CONTEXT_HOST=0.0.0.0 \
    CODE_CONTEXT_PORT=7438 \
    CODE_CONTEXT_DEVICE=cuda \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=4 \
    OPENBLAS_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    MALLOC_ARENA_MAX=2 \
    HF_HOME=/cache/huggingface \
    XDG_CACHE_HOME=/cache/huggingface \
    TORCH_HOME=/cache/huggingface/torch

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src

RUN pip install --no-cache-dir ".[embedding]" \
    && python -c "import torch; v=tuple(map(int, torch.version.cuda.split('.')[:2])); assert v >= (13, 4), torch.version.cuda" \
    && (getent group 10001 >/dev/null || groupadd --system --gid 10001 codecontext) \
    && (getent passwd 10001 >/dev/null || useradd --system --uid 10001 --gid 10001 --home-dir /nonexistent --shell /usr/sbin/nologin codecontext) \
    && mkdir -p /data /cache/huggingface \
    && chown -R 10001:10001 /data /cache

USER 10001:10001

EXPOSE 7438

HEALTHCHECK --interval=30s --timeout=3s --start-period=60s --retries=3 \
  CMD python -c "import os,socket; s=socket.create_connection(('127.0.0.1',int(os.environ.get('CODE_CONTEXT_PORT','7438'))),2); s.close()" || exit 1

CMD ["python", "-m", "code_context_mcp.mps_entrypoint"]
