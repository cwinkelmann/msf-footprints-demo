#!/usr/bin/env bash
# Run footprint extraction on one GeoTIFF.
#
#   ./run.sh <input.tif> <output_dir> [run.py options...]
#   ./run.sh aoi.tif out/ --context arid_urban --bbox 708000 409500 709000 410500
#
# Env: IMAGE (default msf-footprints:0.1), MODEL_DIR (mount a run dir over the baked-in one),
#      GPU=0 to force CPU. Input is mounted read-only; nothing else leaves the machine.
set -euo pipefail
[ $# -ge 2 ] || { sed -n 2,8p "$0"; exit 1; }
IMAGE="${IMAGE:-msf-footprints:0.1}"
IN="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
OUT="$2"; shift 2
mkdir -p "$OUT"; OUT="$(cd "$OUT" && pwd)"

GPU_ARGS=()
if [ "${GPU:-1}" != "0" ] && docker info 2>/dev/null | grep -q -i nvidia; then GPU_ARGS=(--gpus all); fi
MODEL_ARGS=()
if [ -n "${MODEL_DIR:-}" ]; then MODEL_ARGS=(-v "$(cd "$MODEL_DIR" && pwd):/models/seghead_joint_all:ro"); fi

docker run --rm "${GPU_ARGS[@]}" "${MODEL_ARGS[@]}" --shm-size=8g \
  -v "$(dirname "$IN"):/in:ro" -v "$OUT:/out" \
  -u "$(id -u):$(id -g)" \
  "$IMAGE" --image "/in/$(basename "$IN")" --out /out "$@"
echo "outputs in $OUT"
