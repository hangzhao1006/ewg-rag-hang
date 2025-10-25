FROM python:3.12-slim-bookworm

# avoid tz/locale prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV PYTHONUNBUFFERED=1

# we need build tools because scikit-learn / lxml etc may compile things
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# copy only requirements first (better layer caching)
COPY requirements.txt /app/requirements.txt

# install deps
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# now copy source code (backend etc)
COPY backend /app/backend
COPY input-datasets /app/input-datasets
COPY secrets /app/secrets
COPY infra/docker-entrypoint.sh /app/docker-entrypoint.sh

# make sure entrypoint is executable
RUN chmod +x /app/docker-entrypoint.sh

# optional: create a convenience symlink inside the image layer
# (this way, even if you run as non-root later, /input-datasets will already exist)
RUN ln -s /app/input-datasets /input-datasets || true

ENTRYPOINT ["/bin/bash", "/app/docker-entrypoint.sh"]
