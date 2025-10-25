# -------------------------------------------------------------------
# 1. Base image
# -------------------------------------------------------------------
FROM python:3.12-slim

# 不要把 apt 的 man pages 警告弄爆屏
RUN set -ex; \
    for i in $(seq 1 8); do mkdir -p "/usr/share/man/man${i}"; done && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      build-essential \
      curl \
      git \
      nano \
      ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# -------------------------------------------------------------------
# 2. Create non-root user
# -------------------------------------------------------------------
ARG APP_USER=app
RUN useradd -m ${APP_USER}

WORKDIR /app

# -------------------------------------------------------------------
# 3. Copy requirement files first (for Docker layer cache)
# -------------------------------------------------------------------
COPY --chown=${APP_USER}:${APP_USER} requirements.txt /app/requirements.txt

# -------------------------------------------------------------------
# 4. Install python deps
# -------------------------------------------------------------------
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt

# -------------------------------------------------------------------
# 5. Copy source code & data (except chroma_index)
# -------------------------------------------------------------------
COPY --chown=${APP_USER}:${APP_USER} backend /app/backend
COPY --chown=${APP_USER}:${APP_USER} input-datasets /app/input-datasets
COPY --chown=${APP_USER}:${APP_USER} secrets /app/secrets

# 我们不在 build 阶段 COPY chroma_index
# 因为第一次可能还没有这个目录。运行容器时用 -v 绑定就好。

# -------------------------------------------------------------------
# 6. entrypoint 脚本（可选）
# -------------------------------------------------------------------
COPY --chown=${APP_USER}:${APP_USER} infra/docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod +x /app/docker-entrypoint.sh

# -------------------------------------------------------------------
# 7. Security: drop to non-root
# -------------------------------------------------------------------
USER ${APP_USER}

# -------------------------------------------------------------------
# 8. Default entry
# -------------------------------------------------------------------
ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["bash"]
