# REPO_ROOT=$(cd .. && pwd)
# export IMAGE_NAME=ewg-data-chat
# export SECRETS_DIR="$REPO_ROOT/secrets"
# export DATA_DIR="$REPO_ROOT/input-datasets"
# export BACKEND_DIR="$REPO_ROOT/backend"
#
# # export BASE_DIR=$(pwd)
# # export SECRETS_DIR=$(pwd)/../secrets/
# export GCS_BUCKET_URI="gs://ewg-data" [REPLACE WITH YOUR BUCKET NAME]
# export GCP_PROJECT="llm-service-accoun" [REPLACE WITH YOUR PROJECT]
# SECRET_NAME="ewg-data"
# # export OPENAI_API_KEY=$(pwd)/../openkey/
#
# # 获取最新版本的 Secret 内容并保存到本地临时文件
# gcloud secrets versions access latest --secret=$SECRET_NAME > "$SECRETS_DIR/model-trainer.json"
#
#
#
# <!-- mlproject01-207413 -->
#
# # Build the image based on the Dockerfile
# docker build -t $IMAGE_NAME -f infra/Dockerfile "$REPO_ROOT"
# # M1/2 chip macs use this line
# # docker build -t $IMAGE_NAME --platform=linux/arm64/v8 -f Dockerfile .
# #
#
# # Run Container
# docker run --rm --name $IMAGE_NAME -ti \
# -v "$BACKEND_DIR":/app/backend \
# -v "$DATA_DIR":/input-datasets \
# -v "$SECRETS_DIR":/secrets \
# -v "$REPO_ROOT/chroma_index":/chroma_index \
# -e GOOGLE_APPLICATION_CREDENTIALS=/secrets/ewg-data.json \
# -e GCP_PROJECT=$GCP_PROJECT \
# -e GCS_BUCKET_URI=$GCS_BUCKET_URI \
# -e OPENAI_API_KEY=$OPENAI_API_KEY \
# $IMAGE_NAME
# gsutil cp ./ewg_face_full.jsonl "${GCS_BUCKET_URI}"
#
# rm "$SECRETS_DIR/model-trainer.json"


#!/bin/bash
set -e

########################################
# 0. 基本路径和变量
########################################

# 当前脚本在 infra/ 目录下
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

IMAGE_NAME="ewg-data-chat"

SECRETS_DIR="$REPO_ROOT/secrets"
DATA_DIR="$REPO_ROOT/input-datasets"
BACKEND_DIR="$REPO_ROOT/backend"
CHROMA_DIR="$REPO_ROOT/chroma_index"

RAW_DIR="$DATA_DIR/raw"
STRUCTURED_DIR="$DATA_DIR/structured"
BOOKS_DIR="$DATA_DIR/books"
OUTPUTS_DIR="$DATA_DIR/outputs"

# GCP 相关
GCS_BUCKET_URI="gs://ewg-data"        # TODO: 改成你真实的 bucket
GCP_PROJECT="llm-service-accoun"      # TODO: 改成你真实的 GCP project id
SECRET_NAME="ewg-data"                # 你在 Secret Manager 里的 secret 名字

# 把 infra/.env 里的 OPENAI_API_KEY 等变量读进来
# 确保 infra/.env 里已经有:
# OPENAI_API_KEY=sk-xxxx
source "$SCRIPT_DIR/.env"

########################################
# 1. 创建本地需要的目录
########################################

mkdir -p "$SECRETS_DIR"
mkdir -p "$RAW_DIR" "$STRUCTURED_DIR" "$BOOKS_DIR" "$OUTPUTS_DIR"
mkdir -p "$CHROMA_DIR"

########################################
# 2. 从 Secret Manager 拉取 service account key
########################################

# 取最新版本的 service account key，写到 secrets/model-trainer.json
gcloud secrets versions access latest --secret="$SECRET_NAME" > "$SECRETS_DIR/model-trainer.json"

# 用这个 key 在宿主机上激活账号，这样我们宿主机可以跑 gsutil
gcloud auth activate-service-account --key-file="$SECRETS_DIR/model-trainer.json" --project "$GCP_PROJECT"

########################################
# 3. 从 GCS bucket 下载原始数据到本地 raw/
########################################

# 假设 bucket 里有 raw/ewg_face_full.jsonl
gsutil cp "$GCS_BUCKET_URI/raw/ewg_face_full.jsonl" "$RAW_DIR/ewg_face_full.jsonl"

########################################
# 4. Build Docker 镜像
########################################

# build context 用 repo 根目录，让 Dockerfile 能 COPY backend/、pyproject.toml 等
docker build -t "$IMAGE_NAME" -f "$SCRIPT_DIR/Dockerfile" "$REPO_ROOT"

########################################
# 5. 启动容器 (交互式shell开发模式)
########################################
# 注意：我们把宿主机的各个目录 mount 到容器固定路径
#   /app/backend         -> 你的代码
#   /input-datasets      -> 原始/清洗/转换/输出
#   /secrets             -> service account key
#   /chroma_index        -> 向量库持久化

docker run --rm --name "$IMAGE_NAME" -ti \
  -v "$BACKEND_DIR":/app/backend \
  -v "$DATA_DIR":/input-datasets \
  -v "$SECRETS_DIR":/secrets \
  -v "$CHROMA_DIR":/chroma_index \
  -e GOOGLE_APPLICATION_CREDENTIALS=/secrets/ewg-data.json \
  -e GCP_PROJECT="$GCP_PROJECT" \
  -e GCS_BUCKET_URI="$GCS_BUCKET_URI" \
  -e OPENAI_API_KEY="$OPENAI_API_KEY" \
  "$IMAGE_NAME"

########################################
# structured jsonl / csv
gsutil cp "$STRUCTURED_DIR/*" "$GCS_BUCKET_URI/structured/"

# 生成的 txt
gsutil cp "$BOOKS_DIR/*" "$GCS_BUCKET_URI/txt/"

# chunks / embeddings
gsutil cp "$OUTPUTS_DIR/*" "$GCS_BUCKET_URI/outputs/"

# chroma index (向量库)
gsutil cp -r "$CHROMA_DIR/*" "$GCS_BUCKET_URI/chroma_index/"

# 6. 清理本地明文 key
########################################
rm "$SECRETS_DIR/model-trainer.json"
