#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")" || exit 1
export PYTHONPATH=src

python src/main.py \
    --surrogate resnet50 \
    --transfer-model all \
    --objective ce \
    --penalty-norm none \
    --run-name resnet50_ce_none_pixel
