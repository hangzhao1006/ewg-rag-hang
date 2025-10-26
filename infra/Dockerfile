# 1. Base image: python 3.12 slim
FROM python:3.12-slim AS base

# 2. We'll create a non-root user "app"
ARG APP_USER=app

# To avoid tz + debconf noise
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    UV_PROJECT_ENV=/.venv

# Optional: set default envs the code expects
ENV INPUT_FOLDER=/app/input-datasets \
    OUTPUT_FOLDER=/app/input-datasets/outputs \
    FULL_ORIGINAL_JSONL=/app/input-datasets/raw/ewg_face_full.jsonl \
    CHROMADB_HOST=chromadb \
    CHROMADB_PORT=8000 \
    EMBEDDING_DIMENSION=256

# 设置 PATH，让 .venv 的 Python 优先
ENV PATH="/.venv/bin:$PATH"

# 3. Install system deps (build tools, curl, etc.) and create user/app dirs
RUN set -ex; \
    for i in $(seq 1 8); do mkdir -p "/usr/share/man/man${i}"; done && \
    apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        curl \
        build-essential \
        git \
        nano \
        ca-certificates \
        && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir uv && \
    useradd -ms /bin/bash ${APP_USER} -d /home/${APP_USER} -u 1000 && \
    mkdir -p /app && chown ${APP_USER}:${APP_USER} /app && \
    mkdir -p /.venv && chown ${APP_USER}:${APP_USER} /.venv

# 4. Switch to non-root for safety
USER ${APP_USER}
WORKDIR /app

# 5. Copy dependency definitions first (to cache deps layer)
COPY --chown=${APP_USER}:${APP_USER} pyproject.toml ./

# 6. Install python deps into /.venv using uv
RUN uv sync --frozen || uv sync

# 7. Copy source code + data scaffolding
COPY --chown=${APP_USER}:${APP_USER} backend /app/backend
COPY --chown=${APP_USER}:${APP_USER} input-datasets /app/input-datasets

# 8. Create directories that will be mounted at runtime
RUN mkdir -p /app/secrets /app/chroma_index

# 9. default entrypoint -> interactive shell
CMD ["/bin/bash", "-c", "source /.venv/bin/activate && exec /bin/bash"]
