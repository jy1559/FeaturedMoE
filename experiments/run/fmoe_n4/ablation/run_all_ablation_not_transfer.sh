#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DEFAULT_ARGS=(
    --setting-scope all
    --setting-tier essential_extended
    --only-stage routing,stage,cue,objective
    --num-base-per-dataset 5
    --seeds 1,2
    --max-evals 10
    --tune-epochs 100
    --tune-patience 10
    --lr-mode band9
)

bash "${SCRIPT_DIR}/ablation_core_global_queue.sh" "${DEFAULT_ARGS[@]}" "$@"
