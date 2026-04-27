#!/usr/bin/env bash
# Q6 Phase-2 standalone: rerun best separation configs with diag + case-eval.
# Use this if Phase-2 failed or you want to rerun it separately after Phase-1.
#
# Usage:
#   bash q6_rerun_diag.sh [--skip-case-eval] [extra args...]
set -euo pipefail
cd /workspace/FeaturedMoE
/venv/FMoE/bin/python experiments/run/final_experiment/real_final_ablation/q6_route_separation.py \
    --datasets KuaiRecLargeStrictPosV2_0.2,beauty,foursquare,retail_rocket,movielens1m,lastfm0.03 \
    --models featured_moe_n3 \
    --top-k-configs 6 \
    --seeds 1 \
    --gpus "${GPUS:-0,1,2,3,4,5,6,7}" \
    --max-run-hours 1.5 \
    --tune-epochs 100 \
    --tune-patience 10 \
    --top-k-diag 2 \
    --phase2-only \
    "$@"
