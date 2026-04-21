#!/usr/bin/env bash
set -euo pipefail
cd /workspace/FeaturedMoE
/venv/FMoE/bin/python experiments/run/final_experiment/real_final_ablation/q4_efficiency.py \
	--datasets "${Q4_DATASETS:-KuaiRecLargeStrictPosV2_0.2,foursquare,lastfm0.03}" \
	--benchmark-datasets "${Q4_BENCHMARK_DATASETS:-KuaiRecLargeStrictPosV2_0.2,foursquare}" \
	--screen-epochs "${Q4_SCREEN_EPOCHS:-8}" \
	--confirm-epochs "${Q4_CONFIRM_EPOCHS:-100}" \
	--train-batch-size "${Q4_TRAIN_BATCH_SIZE:-512}" \
	--eval-batch-size "${Q4_EVAL_BATCH_SIZE:-1024}" \
	--include-fame \
	--lr-multipliers "${Q4_LR_MULTIPLIERS:-0.8,1.0}" \
	--confirm-lr-multipliers "${Q4_CONFIRM_LR_MULTIPLIERS:-0.8,1.0,1.2}" \
	--width-multipliers "${Q4_WIDTH_MULTIPLIERS:-0.5,0.75,1.0,1.25}" \
	--group-topk-grid "${Q4_GROUP_TOPK_GRID:-1,2,3,4}" \
	--expert-topk-grid "${Q4_EXPERT_TOPK_GRID:-1,2}" \
	--max-route-screen-runs "${Q4_MAX_ROUTE_SCREEN_RUNS:-8}" \
	--active-match-tolerance "${Q4_ACTIVE_MATCH_TOLERANCE:-0.15}" \
	"$@"
