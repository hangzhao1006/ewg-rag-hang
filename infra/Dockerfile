# docker build -t ewg-data-chat -f infra/Dockerfile .

# Use the official Debian-hosted Python image
FROM python:3.12-slim-bookworm

ARG DEBIAN_PACKAGES="build-essential curl libpq5"
# libpq-dev

# Prevent apt from showing prompts
ENV DEBIAN_FRONTEND=noninteractive

# Python wants UTF-8 locale
ENV LANG=C.UTF-8

# Tell Python to disable buffering so we don't lose any logs.
ENV PYTHONUNBUFFERED=1

# Tell uv to copy packages from the wheel into the site-packages
ENV UV_LINK_MODE=copy
ENV UV_PROJECT_ENVIRONMENT=/.venv

# Ensure we have an up to date baseline, install dependencies and
# create a user so we don't run the app as root
RUN set -ex; \
    for i in $(seq 1 8); do mkdir -p "/usr/share/man/man${i}"; done && \
    apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends $DEBIAN_PACKAGES && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir --upgrade pip && \
    pip install uv && \
    useradd -ms /bin/bash app -d /home/app -u 1000 && \
    mkdir -p /app && \
    chown app:app /app && \
    mkdir -p /.venv && \
    chown app:app /.venv

# Switch to the new user
USER app
WORKDIR /app

# after setting USER app and WORKDIR /app

# 先复制依赖声明
COPY --chown=app:app pyproject.toml uv.lock* requirements.txt /app/

# 安装依赖
RUN uv sync --frozen || pip install -r requirements.txt

# 复制代码
COPY --chown=app:app backend /app/backend
COPY --chown=app:app input-datasets /app/input-datasets
COPY --chown=app:app infra/docker-entrypoint.sh /app/docker-entrypoint.sh
COPY --chown=app:app secrets /app/secrets

# 设定入口点（开发期可以继续给你一个交互shell）
ENTRYPOINT ["/bin/bash", "/app/docker-entrypoint.sh"]
