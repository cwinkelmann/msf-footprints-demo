#!/usr/bin/env bash
# Build the image. Put a model run dir under ./models/ first to bake it in (see README).
set -euo pipefail
cd "$(dirname "$0")"
IMAGE="${IMAGE:-msf-footprints:0.1}"
docker build -t "$IMAGE" .
echo "built $IMAGE"
