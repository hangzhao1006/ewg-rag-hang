export IMAGE_NAME=ewg-data-chat
export BASE_DIR=$(pwd)
export SECRETS_DIR=$(pwd)/../secrets/
export GCS_BUCKET_URI="gs://ewg-data" [REPLACE WITH YOUR BUCKET NAME]
export GCP_PROJECT="llm-service-accoun" [REPLACE WITH YOUR PROJECT]
# export OPENAI_API_KEY=$(pwd)/../openkey/

<!-- mlproject01-207413 -->

# Build the image based on the Dockerfile
docker build -t $IMAGE_NAME -f infra/Dockerfile .
# M1/2 chip macs use this line
# docker build -t $IMAGE_NAME --platform=linux/arm64/v8 -f Dockerfile .

# Run Container
docker run --rm --name $IMAGE_NAME -ti \
--env-file .env \
-v "$BASE_DIR":/app \
-v "$SECRETS_DIR":/secrets \
-e GOOGLE_APPLICATION_CREDENTIALS=/secrets/model-trainer.json \
-e GCP_PROJECT=$GCP_PROJECT \
-e GCS_BUCKET_URI=$GCS_BUCKET_URI \
-e OPENAI_API_KEY=$OPENAI_API_KEY \
$IMAGE_NAME
