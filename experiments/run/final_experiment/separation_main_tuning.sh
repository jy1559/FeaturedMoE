#!/usr/bin/env bash
set -euo pipefail

cd /workspace/FeaturedMoE
/venv/FMoE/bin/python experiments/run/final_experiment/separation_main_tuning.py \
    --datasets KuaiRecLargeStrictPosV2_0.2,foursquare \
    --models featured_moe_n3 \
    --top-k-configs 4 \
    --seeds 1 \
    --gpus "${GPUS:-0,1,2,3,4,5,6,7}" \
    --max-evals 30 \
    --max-run-hours 3.0 \
    --tune-epochs 100 \
    --tune-patience 10 \
    --search-algo tpe \
    --resume-from-logs \
    "$@"