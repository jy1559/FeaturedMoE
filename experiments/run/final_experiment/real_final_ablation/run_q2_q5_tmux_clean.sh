#!/usr/bin/env bash
set -euo pipefail

# Clean + rerun Q2~Q5 with per-question dataset/seed/hparam budgets.
# Intended for an already-running tmux session.

REPO_ROOT="/workspace/FeaturedMoE"
PY_BIN="${RUN_PYTHON_BIN:-/venv/FMoE/bin/python}"
Q_DIR="${REPO_ROOT}/experiments/run/final_experiment/real_final_ablation"

# ----- Shared runtime knobs -----
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
BASE_CSV="${BASE_CSV:-${REPO_ROOT}/experiments/run/final_experiment/ablation/configs/base_candidates.csv}"
SEARCH_ALGO="${SEARCH_ALGO:-tpe}"
LR_MODE="${LR_MODE:-narrow_loguniform}"
TUNE_EPOCHS="${TUNE_EPOCHS:-100}"
TUNE_PATIENCE="${TUNE_PATIENCE:-10}"
DRY_RUN_FLAG="${DRY_RUN_FLAG:-}"

# ----- Q2 (multi-dataset; split lastfm profile) -----
Q2_MAIN_DATASETS="${Q2_MAIN_DATASETS:-KuaiRecLargeStrictPosV2_0.2,retail_rocket,foursquare}"
Q2_LFM_DATASET="${Q2_LFM_DATASET:-lastfm0.03}"
Q2_TOPK="${Q2_T OPK:-2}"
Q2_SEEDS="${Q2_SEEDS:-1,2,3}"
Q2_MAX_EVALS="${Q2_MAX_EVALS:-4}"
Q2_LFM_SEEDS="${Q2_LFM_SEEDS:-1,2}"
Q2_LFM_MAX_EVALS="${Q2_LFM_MAX_EVALS:-2}"
Q2_MAX_RUN_HOURS="${Q2_MAX_RUN_HOURS:-1.2}"

# ----- Q3~Q5 (selected datasets) -----
Q345_DATASET="${Q345_DATASET:-KuaiRecLargeStrictPosV2_0.2,foursquare}"
Q345_TOPK="${Q345_TOPK:-3}"
Q345_SEEDS="${Q345_SEEDS:-1,2,3}"
Q3_MAX_EVALS="${Q3_MAX_EVALS:-6}"
Q4_MAX_EVALS="${Q4_MAX_EVALS:-4}"
Q5_MAX_EVALS="${Q5_MAX_EVALS:-5}"
Q345_MAX_RUN_HOURS="${Q345_MAX_RUN_HOURS:-1.2}"

# ----- Behavior flags -----
CLEAN_FIRST="${CLEAN_FIRST:-1}"
RESUME_FLAG="${RESUME_FLAG:---no-resume-from-logs}"
CASE_EVAL_FAST_FLAG="${CASE_EVAL_FAST_FLAG:-}"
POSTPROCESS_ALL_FLAG="${POSTPROCESS_ALL_FLAG:-}"
Q2_EXTRA_ARGS="${Q2_EXTRA_ARGS:-}"
Q3_EXTRA_ARGS="${Q3_EXTRA_ARGS:-}"
Q4_EXTRA_ARGS="${Q4_EXTRA_ARGS:-}"
Q5_EXTRA_ARGS="${Q5_EXTRA_ARGS:-}"

cd "${REPO_ROOT}"

if [[ "${CLEAN_FIRST}" == "1" ]]; then
  echo "[clean] removing old q2~q5 logs and q5 postprocess artifacts"
  rm -rf experiments/run/artifacts/logs/real_final_ablation/q2
  rm -rf experiments/run/artifacts/logs/real_final_ablation/q3
  rm -rf experiments/run/artifacts/logs/real_final_ablation/q4
  rm -rf experiments/run/artifacts/logs/real_final_ablation/q5
  rm -rf experiments/run/artifacts/logs/real_final_ablation/_tmp_case_eval_validation
  rm -rf experiments/run/artifacts/logs/real_final_ablation/_tmp_case_eval_pipeline_validation
fi

echo "[run] Q2 routing control"
"${PY_BIN}" "${Q_DIR}/q2_routing_control.py" \
  --datasets "${Q2_MAIN_DATASETS}" \
  --models featured_moe_n3 \
  --top-k-configs "${Q2_TOPK}" \
  --seeds "${Q2_SEEDS}" \
  --gpus "${GPU_LIST}" \
  --base-csv "${BASE_CSV}" \
  --max-evals "${Q2_MAX_EVALS}" \
  --max-run-hours "${Q2_MAX_RUN_HOURS}" \
  --tune-epochs "${TUNE_EPOCHS}" \
  --tune-patience "${TUNE_PATIENCE}" \
  --lr-mode "${LR_MODE}" \
  --search-algo "${SEARCH_ALGO}" \
  ${RESUME_FLAG} \
  ${DRY_RUN_FLAG} \
  ${Q2_EXTRA_ARGS}

echo "[run] Q2 routing control (lastfm budget)"
"${PY_BIN}" "${Q_DIR}/q2_routing_control.py" \
  --datasets "${Q2_LFM_DATASET}" \
  --models featured_moe_n3 \
  --top-k-configs "${Q2_TOPK}" \
  --seeds "${Q2_LFM_SEEDS}" \
  --gpus "${GPU_LIST}" \
  --base-csv "${BASE_CSV}" \
  --max-evals "${Q2_LFM_MAX_EVALS}" \
  --max-run-hours "${Q2_MAX_RUN_HOURS}" \
  --tune-epochs "${TUNE_EPOCHS}" \
  --tune-patience "${TUNE_PATIENCE}" \
  --lr-mode "${LR_MODE}" \
  --search-algo "${SEARCH_ALGO}" \
  ${RESUME_FLAG} \
  ${DRY_RUN_FLAG} \
  ${Q2_EXTRA_ARGS}

echo "[run] Q3 stage structure"
"${PY_BIN}" "${Q_DIR}/q3_stage_structure.py" \
  --datasets "${Q345_DATASET}" \
  --models featured_moe_n3 \
  --top-k-configs "${Q345_TOPK}" \
  --seeds "${Q345_SEEDS}" \
  --gpus "${GPU_LIST}" \
  --base-csv "${BASE_CSV}" \
  --max-evals "${Q3_MAX_EVALS}" \
  --max-run-hours "${Q345_MAX_RUN_HOURS}" \
  --tune-epochs "${TUNE_EPOCHS}" \
  --tune-patience "${TUNE_PATIENCE}" \
  --lr-mode "${LR_MODE}" \
  --search-algo "${SEARCH_ALGO}" \
  ${RESUME_FLAG} \
  ${DRY_RUN_FLAG} \
  ${Q3_EXTRA_ARGS}

echo "[run] Q4 efficiency"
"${PY_BIN}" "${Q_DIR}/q4_efficiency.py" \
  --datasets "${Q345_DATASET}" \
  --models featured_moe_n3 \
  --top-k-configs "${Q345_TOPK}" \
  --seeds "${Q345_SEEDS}" \
  --gpus "${GPU_LIST}" \
  --base-csv "${BASE_CSV}" \
  --max-evals "${Q4_MAX_EVALS}" \
  --max-run-hours "${Q345_MAX_RUN_HOURS}" \
  --tune-epochs "${TUNE_EPOCHS}" \
  --tune-patience "${TUNE_PATIENCE}" \
  --lr-mode "${LR_MODE}" \
  --search-algo "${SEARCH_ALGO}" \
  ${RESUME_FLAG} \
  ${DRY_RUN_FLAG} \
  ${Q4_EXTRA_ARGS}

echo "[run] Q5 behavior semantics"
"${PY_BIN}" "${Q_DIR}/q5_behavior_semantics.py" \
  --datasets "${Q345_DATASET}" \
  --models featured_moe_n3 \
  --top-k-configs "${Q345_TOPK}" \
  --seeds "${Q345_SEEDS}" \
  --gpus "${GPU_LIST}" \
  --base-csv "${BASE_CSV}" \
  --max-evals "${Q5_MAX_EVALS}" \
  --max-run-hours "${Q345_MAX_RUN_HOURS}" \
  --tune-epochs "${TUNE_EPOCHS}" \
  --tune-patience "${TUNE_PATIENCE}" \
  --lr-mode "${LR_MODE}" \
  --search-algo "${SEARCH_ALGO}" \
  ${RESUME_FLAG} \
  ${DRY_RUN_FLAG} \
  ${CASE_EVAL_FAST_FLAG} \
  ${POSTPROCESS_ALL_FLAG} \
  ${Q5_EXTRA_ARGS}

echo "[done] Q2~Q5 complete"
