#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG=${1:?config path is required}
CHECKPOINT=${2:?checkpoint path is required}
export PORT=${PORT:-29500}

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3} \
  bash tools/dist_test.sh "$CONFIG" "$CHECKPOINT" 4 "${@:3}"
