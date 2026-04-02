#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-finmem-dev}"
CONTAINER_NAME="${CONTAINER_NAME:-finmem-drl}"
HOST_PORT="${HOST_PORT:-8001}"

docker run -it --rm \
  --platform linux/amd64 \
  --name "${CONTAINER_NAME}" \
  -p "${HOST_PORT}:8000" \
  -v "$(pwd):/finmem" \
  -w /finmem \
  "${IMAGE_NAME}" \
  bash
