#!/usr/bin/env bash
# Q6: route_consistency + route_separation joint hyperopt (Phase-1) + diag rerun (Phase-2).
#
# Phase-1: 6 datasets × top-6 base configs = 36 jobs, 8 GPUs
#   Each job tunes LR (tight), route_consistency_lambda, route_separation_lambda via TPE.
# Phase-2: per-dataset top-2 best results rerun with diag + case-eval
#
# Usage:
#   bash q6_route_separation.sh [--dry-run] [--smoke-test] [extra args...]
#
# Override GPUs:
#   GPUS=0,1,2,3,4,5,6,7 bash q6_route_separation.sh
#
# Skip Phase-2:
#   bash q6_route_separation.sh --skip-phase2
#
# Skip case-eval in Phase-2:
#   bash q6_route_separation.sh --skip-case-eval
set -euo pipefail
cd /workspace/FeaturedMoE
/venv/FMoE/bin/python experiments/run/final_experiment/real_final_ablation/q6_route_separation.py \
    --datasets KuaiRecLargeStrictPosV2_0.2,beauty,foursquare,retail_rocket,movielens1m,lastfm0.03 \
    --models featured_moe_n3 \
    --top-k-configs 6 \
    --seeds 1 \
    --gpus "${GPUS:-0,1,2,3,4,5,6,7}" \
    --max-evals 15 \
    --max-run-hours 1.5 \
    --tune-epochs 100 \
    --tune-patience 10 \
    --lr-mode narrow_loguniform \
    --search-algo tpe \
    --resume-from-logs \
    --top-k-diag 2 \
    "$@"
