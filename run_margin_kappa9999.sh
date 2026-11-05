#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")" || exit 1
export PYTHONPATH=src

pids=()
for norm in l1 l2 paper none; do
    name="resnet50_margin_kappa9999_${norm}_pixel"
    python src/main.py \
        --surrogate resnet50 \
        --transfer-model all \
        --objective margin \
        --kappa 9999 \
        --penalty-norm "$norm" \
        --run-name "$name" &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
exit "$status"
