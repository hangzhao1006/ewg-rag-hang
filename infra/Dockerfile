# 1. 基础镜像：官方 python 3.12 slim
FROM python:3.12-slim

# 2. 让我们用一个非root用户运行最终进程（安全一点）
ARG APP_USER=app
RUN useradd -m ${APP_USER}

WORKDIR /app

# 3. 安装系统依赖，确保 scikit-learn / numpy / etc 可以编译&运行
#    还有 git 以防某些依赖需要
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# 4. 复制 requirements.txt 并安装 python 依赖
#    我们先复制 requirements.txt，这样如果代码改了但依赖没改，
#    docker layer 可以复用缓存
COPY --chown=${APP_USER}:${APP_USER} requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt

# 5. 复制代码和数据结构
#    这些目录名基于你repo的 layout:
#    backend/, input-datasets/, secrets/ (空壳也行), chroma_index/ (可选)
COPY --chown=${APP_USER}:${APP_USER} backend /app/backend
COPY --chown=${APP_USER}:${APP_USER} input-datasets /app/input-datasets
COPY --chown=${APP_USER}:${APP_USER} secrets /app/secrets
COPY --chown=${APP_USER}:${APP_USER} chroma_index /app/chroma_index 2>/dev/null || true

# 6. 把 entrypoint 脚本（如果你有 infra/docker-entrypoint.sh）
COPY --chown=${APP_USER}:${APP_USER} infra/docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod +x /app/docker-entrypoint.sh

# 7. 切到非root用户运行
USER ${APP_USER}

# 8. 默认工作目录
WORKDIR /app

# 9. 默认入口（可以保留你原来的逻辑，比如打印环境等）
ENTRYPOINT ["/app/docker-entrypoint.sh"]
