#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG=${1:?config path is required}
WORK_DIR=${2:?work directory is required}

python tools/train.py "$CONFIG" --work-dir "$WORK_DIR" "${@:3}"
