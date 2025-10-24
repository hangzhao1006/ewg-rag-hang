#!/bin/bash
set -e

# 1. 先把 repo root 计算出来
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

SECRETS_DIR="$REPO_ROOT/secrets"
mkdir -p "$SECRETS_DIR"

SECRET_NAME="ewg-data"

# 2. 从 Secret Manager 拉 service account key
gcloud secrets versions access latest --secret="$SECRET_NAME" > "$SECRETS_DIR/ewg-data.json"

# 3. 启动 compose
cd "$SCRIPT_DIR"
docker compose up --build -d

# (可选) 不立刻删 ewg-data.json，因为容器还在用这个 volume。
# 如果你删掉，本地 volume 也没了，容器重启就会炸。
