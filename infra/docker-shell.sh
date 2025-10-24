REPO_ROOT=$(cd .. && pwd)
export IMAGE_NAME=ewg-data-chat
export SECRETS_DIR="$REPO_ROOT/secrets"
export DATA_DIR="$REPO_ROOT/input-datasets"
export BACKEND_DIR="$REPO_ROOT/backend"

# export BASE_DIR=$(pwd)
# export SECRETS_DIR=$(pwd)/../secrets/
export GCS_BUCKET_URI="gs://ewg-data" [REPLACE WITH YOUR BUCKET NAME]
export GCP_PROJECT="llm-service-accoun" [REPLACE WITH YOUR PROJECT]
SECRET_NAME="ewg-data"
# export OPENAI_API_KEY=$(pwd)/../openkey/

# 获取最新版本的 Secret 内容并保存到本地临时文件
gcloud secrets versions access latest --secret=$SECRET_NAME > "$SECRETS_DIR/model-trainer.json"



<!-- mlproject01-207413 -->

# Build the image based on the Dockerfile
docker build -t $IMAGE_NAME -f infra/Dockerfile "$REPO_ROOT"
# M1/2 chip macs use this line
# docker build -t $IMAGE_NAME --platform=linux/arm64/v8 -f Dockerfile .

# Run Container
docker run --rm --name $IMAGE_NAME -ti \
-v "$BACKEND_DIR":/app/backend \
-v "$DATA_DIR":/input-datasets \
-v "$SECRETS_DIR":/secrets \
-v "$REPO_ROOT/chroma_index":/chroma_index \
-e GOOGLE_APPLICATION_CREDENTIALS=/secrets/ewg-data.json \
-e GCP_PROJECT=$GCP_PROJECT \
-e GCS_BUCKET_URI=$GCS_BUCKET_URI \
-e OPENAI_API_KEY=$OPENAI_API_KEY \
$IMAGE_NAME

rm "$SECRETS_DIR/model-trainer.json"
