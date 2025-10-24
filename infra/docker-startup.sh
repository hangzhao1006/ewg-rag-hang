#!/bin/bash

# 使用 gcloud CLI 从 Secret Manager 获取密钥文件
SECRET_NAME="ewg-data"
gcloud secrets versions access 1 --secret=projects/301578946659/secrets/ewg-data > ../secrets/ewg-data.json


# 启动 Docker Compose
docker compose up -d

# 停止后手动清理密钥文件
# 例如，通过 docker exec rm ... 或者在部署脚本中添加清理步骤
