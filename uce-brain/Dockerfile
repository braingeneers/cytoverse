FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl unzip ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# awscli v2 (musl-free, linux-x86_64 or aarch64 resolved at build time)
RUN set -eu; \
    arch="$(uname -m)"; \
    case "$arch" in \
      x86_64)  url="https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" ;; \
      aarch64) url="https://awscli.amazonaws.com/awscli-exe-linux-aarch64.zip" ;; \
      *) echo "unsupported arch: $arch" >&2; exit 1 ;; \
    esac; \
    curl -fsSL "$url" -o /tmp/awscli.zip; \
    unzip -q /tmp/awscli.zip -d /tmp; \
    /tmp/aws/install; \
    rm -rf /tmp/aws /tmp/awscli.zip

# uv for fast deps install
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Deps first (cache layer)
COPY pyproject.toml uv.lock /app/
RUN uv sync --frozen --no-install-project

# Project source (no editable install; we run scripts directly)
COPY scripts /app/scripts
COPY uce-brain /app/uce-brain
COPY data/models /app/data/models

ENV PATH="/app/.venv/bin:${PATH}"

ENTRYPOINT ["/bin/bash", "-c"]
