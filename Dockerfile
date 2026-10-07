FROM python:3.12-slim-bookworm

ARG INSTALL_EMBEDDING=1

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CODE_CONTEXT_HOME=/data \
    CODE_CONTEXT_HOST=0.0.0.0 \
    CODE_CONTEXT_PORT=7438 \
    CODE_CONTEXT_DEVICE=cpu \
    NVIDIA_VISIBLE_DEVICES=void \
    CUDA_VISIBLE_DEVICES="" \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=4 \
    OPENBLAS_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    MALLOC_ARENA_MAX=2 \
    HF_HOME=/cache/huggingface

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src

RUN if [ "$INSTALL_EMBEDDING" = "1" ]; then \
        pip install --no-cache-dir ".[embedding]"; \
    else \
        pip install --no-cache-dir "."; \
    fi \
    && groupadd --system --gid 10001 codecontext \
    && useradd --system --uid 10001 --gid 10001 --home-dir /nonexistent --shell /usr/sbin/nologin codecontext \
    && mkdir -p /data /cache/huggingface \
    && chown -R 10001:10001 /data /cache

USER 10001:10001

EXPOSE 7438

HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
  CMD python -c "import os,socket; s=socket.create_connection(('127.0.0.1',int(os.environ.get('CODE_CONTEXT_PORT','7438'))),2); s.close()" || exit 1

CMD ["python", "-m", "code_context_mcp.docker_entrypoint"]
