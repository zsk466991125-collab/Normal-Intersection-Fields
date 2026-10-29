#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG=${1:?config path is required}
WORK_DIR=${2:?work directory is required}
export PORT=${PORT:-29500}

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3} \
  bash tools/dist_train.sh "$CONFIG" 4 \
  --work-dir "$WORK_DIR" \
  --cfg-options data.samples_per_gpu=2 "${@:3}"
