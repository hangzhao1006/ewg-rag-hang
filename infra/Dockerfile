# docker build -t ewg-data-chat -f infra/Dockerfile .

# 基础镜像：Debian bookworm + Python 3.12（保持你的选择）
FROM python:3.12-slim-bookworm

# 我们把需要的系统依赖放这里，方便后面 apt 一次性装
# 说明：
# - build-essential / gcc / g++: build C/C++ extensions (pandas, chromadb deps, etc.)
# - python3-dev: 头文件，某些包会需要
# - libffi-dev, libssl-dev: 加密/ffi依赖 (openai, requests, etc.)
# - libsqlite3-dev: chromadb / sqlite 后端常用
# - curl, git: uv/pip 有时需要下载额外资源
# - rustc, cargo: tiktoken / chromadb 的一部分依赖是 Rust
ARG DEBIAN_PACKAGES="build-essential gcc g++ python3-dev libffi-dev libssl-dev libsqlite3-dev curl git rustc cargo libpq5"

# 避免 apt 交互
ENV DEBIAN_FRONTEND=noninteractive
# Python stdout 不缓存
ENV PYTHONUNBUFFERED=1
# Python locale
ENV LANG=C.UTF-8

# uv 的相关 env（沿用你的设定）
ENV UV_LINK_MODE=copy
ENV UV_PROJECT_ENVIRONMENT=/.venv

# 这里我们仍在 root 身份下：
# - 更新 apt
# - 安装全部系统依赖（包含我们扩展的 build deps）
# - 安装 pip/uv
# - 创建非 root 用户 app，准备工作目录和虚拟环境目录
RUN set -ex; \
    for i in $(seq 1 8); do mkdir -p "/usr/share/man/man${i}"; done && \
    apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends $DEBIAN_PACKAGES && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir uv && \
    useradd -ms /bin/bash app -d /home/app -u 1000 && \
    mkdir -p /app && \
    chown app:app /app && \
    mkdir -p /.venv && \
    chown app:app /.venv

# 切换到非 root 用户
USER app
WORKDIR /app

# 把依赖声明文件放进镜像，以便 uv / pip 能跑
# 注意：我们保留了你的三类依赖声明，外加我们创建的 requirements.txt
# - pyproject.toml / uv.lock* 使 uv 能精确复现
# - requirements.txt 作为 fallback（比如 uv.lock 不完整时）
COPY --chown=app:app pyproject.toml uv.lock* requirements.txt /app/

# 安装 Python 依赖
# 逻辑：
# 1. 先尝试 uv sync --frozen（理想路径，走你的 lock）
# 2. 如果 uv 环境不完整，则 fallback 到 requirements.txt
#
# 重要：我们在前面已经把编译工具链装进去了（build-essential, cargo等）
# 这样 chromadb / tiktoken / pandas / bs4 这些包都应该能编译或拿到轮子
RUN uv sync --frozen || pip install --no-cache-dir -r /app/requirements.txt

# 复制你的项目代码进镜像
# 我们分开 COPY，而不是 COPY . /app，是为了避免把宿主机的一堆没必要的临时东西（例如 .venv、本地缓存）带进镜像层
COPY --chown=app:app backend /app/backend
COPY --chown=app:app input-datasets /app/input-datasets
COPY --chown=app:app secrets /app/secrets
COPY --chown=app:app infra/docker-entrypoint.sh /app/docker-entrypoint.sh

# 入口点保持你的逻辑（交互式开发友好）
ENTRYPOINT ["/bin/bash", "/app/docker-entrypoint.sh"]
