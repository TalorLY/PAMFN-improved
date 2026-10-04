#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source runtime.env
exec "$V38_PYTHON" -u run.py --stage all --gpus "$GPU_LIST"
