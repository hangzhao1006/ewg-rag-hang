FROM python:3.12-slim-bookworm

# we only need lightweight build deps for pandas / chromadb / requests crypto, etc.
ARG DEBIAN_PACKAGES="build-essential gcc g++ python3-dev libffi-dev libssl-dev libsqlite3-dev curl git"

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV LANG=C.UTF-8

# install system deps, upgrade pip, create app user
RUN set -ex; \
    for i in $(seq 1 8); do mkdir -p "/usr/share/man/man${i}"; done && \
    apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends $DEBIAN_PACKAGES && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir --upgrade pip setuptools wheel && \
    useradd -ms /bin/bash app -d /home/app -u 1000 && \
    mkdir -p /app && chown app:app /app

USER app
WORKDIR /app

# copy requirements first for cache
COPY --chown=app:app requirements.txt /app/requirements.txt

# install ALL python deps into the *system* python (no venv)
RUN pip install --no-cache-dir -r /app/requirements.txt

# copy code and data dirs
COPY --chown=app:app backend /app/backend
COPY --chown=app:app input-datasets /app/input-datasets
COPY --chown=app:app secrets /app/secrets
COPY --chown=app:app infra/docker-entrypoint.sh /app/docker-entrypoint.sh

# make sure entrypoint is executable
RUN chmod +x /app/docker-entrypoint.sh

ENTRYPOINT ["/bin/bash", "/app/docker-entrypoint.sh"]

