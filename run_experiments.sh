#!/usr/bin/env bash
# Usage: ./run_experiments.sh [pixel|latent|both]   (default: pixel)
set -uo pipefail

cd "$(dirname "$0")"
export PYTHONPATH=src

mode="${1:-pixel}"
case "$mode" in
    pixel) latent_modes=(0) ;;
    latent) latent_modes=(1) ;;
    both) latent_modes=(0 1) ;;
    *) echo "Usage: $0 [pixel|latent|both]" >&2; exit 2 ;;
esac

pids=()
for latent in "${latent_modes[@]}"; do
    for norm in l1 l2 none; do
        extra=()
        name="resnet50_${norm}"
        if [ "$latent" = 1 ]; then
            extra=(--latent-mode)
            name="${name}_latent"
        fi
        python src/main.py \
            --surrogate resnet50 \
            --transfer-model all \
            --penalty-norm "$norm" \
            --run-name "$name" \
            "${extra[@]}" &
        pids+=($!)
    done
done

status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
exit "$status"
