# infra/Dockerfile

# 1. base image
FROM python:3.12-slim-bookworm

# 2. system deps we actually need
#    - curl just for debugging
#    - build-essential & gcc so that scikit-learn / sentence-transformers wheels can finalize
#    - libgomp1 is often needed for sklearn / numpy performance
RUN set -ex \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
         build-essential \
         gcc \
         curl \
         libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 3. workdir and non-root user
WORKDIR /app
RUN useradd -ms /bin/bash app -u 1000
USER app

# 4. copy just dependency metadata first (for build cache)
#    我们用 pyproject.toml 作为主依赖声明。如果你还保留 requirements.txt 也可以暂时COPY去兼容。
COPY --chown=app:app pyproject.toml /app/pyproject.toml

# 5. install deps
#    我们先升级pip，然后用 pip 根据 pyproject.toml 安装依赖（"pip install ." 需要源代码能import，
#    所以更保险的办法是先用 a minimal requirements lock；这里简单处理：直接用 pip install -r）
#
#    为了简单：我们生成一个轻量 requirements.runtime.txt (可选)。
#    但如果你不想维护多份文件，也可以在 Docker build 之前用:
#       pip install ".[whatever]"
#    这里我们走最直接的：把 pyproject.toml 复制后，用 pip-tools/uv 转成 wheel。在没有这些工具的前提下，
#    我们可以在镜像里直接 `pip install pandas==...` etc.
#
#    ——最简单稳定的做法：同时在 repo 根目录保留一个小的 requirements.txt，对应我们 pyproject 的依赖列表。
#
#    所以：请在仓库根目录新建一个 requirements.txt，内容和 pyproject.toml 里的 dependencies 一致。
COPY --chown=app:app requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt

# 6. copy code (but NOT secrets)
COPY --chown=app:app backend /app/backend
COPY --chown=app:app input-datasets /app/input-datasets
COPY --chown=app:app chroma_index /app/chroma_index

# 你可以选择不COPY secrets，因为运行时我们会用 volume 把宿主机 ./secrets 挂进容器
# COPY --chown=app:app secrets /app/secrets

# entrypoint:
# 我们保持一个可交互的容器，compose 里会用 tty 进入，不强行CMD python。
CMD ["/bin/bash"]
