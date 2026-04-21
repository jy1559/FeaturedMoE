#!/usr/bin/env bash
set -euo pipefail
cd /workspace/FeaturedMoE
/venv/FMoE/bin/python experiments/run/final_experiment/real_final_ablation/q4_portability.py \
	--datasets "${Q4_DATASETS:-KuaiRecLargeStrictPosV2_0.2,foursquare}" \
	--models "${Q4_MODELS:-featured_moe_n3}" \
	--base-csv "${Q4_BASE_CSV:-experiments/run/final_experiment/real_final_ablation/configs/q4_portability_base_candidates.csv}" \
	--top-k-configs "${Q4_TOP_K_CONFIGS:-2}" \
	--seeds "${Q4_SEEDS:-1}" \
	--gpus "${Q4_GPUS:-0}" \
	--max-evals "${Q4_MAX_EVALS:-4}" \
	--max-run-hours "${Q4_MAX_RUN_HOURS:-1.0}" \
	--tune-epochs "${Q4_TUNE_EPOCHS:-100}" \
	--tune-patience "${Q4_TUNE_PATIENCE:-10}" \
	--lr-mode "${Q4_LR_MODE:-narrow_loguniform}" \
	--search-algo "${Q4_SEARCH_ALGO:-tpe}" \
	"$@"
