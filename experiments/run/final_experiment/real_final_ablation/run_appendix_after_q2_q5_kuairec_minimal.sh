#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/workspace/FeaturedMoE"
PY_BIN="${RUN_PYTHON_BIN:-/venv/FMoE/bin/python}"
APPENDIX_DIR="${REPO_ROOT}/experiments/run/final_experiment/real_final_ablation/appendix"

DATASET="${DATASET:-KuaiRecLargeStrictPosV2_0.2}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
BASE_CSV="${BASE_CSV:-${REPO_ROOT}/experiments/run/final_experiment/ablation/configs/base_candidates.csv}"
SEARCH_ALGO="${SEARCH_ALGO:-tpe}"
LR_MODE="${LR_MODE:-narrow_loguniform}"

TOPK="${TOPK:-1}"
SEEDS="${SEEDS:-1}"
MAX_EVALS="${MAX_EVALS:-1}"
MAX_RUN_HOURS="${MAX_RUN_HOURS:-0.6}"
TUNE_EPOCHS="${TUNE_EPOCHS:-100}"
TUNE_PATIENCE="${TUNE_PATIENCE:-10}"
RESUME_FLAG="${RESUME_FLAG:---resume-from-logs}"
POLL_SECONDS="${POLL_SECONDS:-60}"

SECTIONS="${SECTIONS:-full_results,special_bins,structural,sparse,objective,cost,diagnostics,behavior_slices,cases}"
DRY_RUN_FLAG="${DRY_RUN_FLAG:-}"

main_ablation_jobs_running() {
  local matches=""
  matches+="$(pgrep -af 'real_final_ablation/q2_routing_control\.py|real_final_ablation/q3_stage_structure\.py|real_final_ablation/q4_efficiency\.py|real_final_ablation/q5_behavior_semantics\.py' || true)"
  matches+=$'\n'
  matches+="$(pgrep -af 'hyperopt_tune\.py.*--run-group real_final_ablation([[:space:]]|$)' || true)"
  matches="$(printf '%s\n' "${matches}" | sed '/^[[:space:]]*$/d' | grep -v 'run_appendix_after_q2_q5_kuairec_minimal.sh' || true)"
  if [[ -n "${matches}" ]]; then
    echo "${matches}"
    return 0
  fi
  return 1
}

summarize_running_jobs() {
  local text="$1"
  local q2_count q3_count q4_count q5_count hyper_count other_count
  q2_count=$(printf '%s\n' "${text}" | grep -c 'q2_routing_control\.py' || true)
  q3_count=$(printf '%s\n' "${text}" | grep -c 'q3_stage_structure\.py' || true)
  q4_count=$(printf '%s\n' "${text}" | grep -c 'q4_efficiency\.py' || true)
  q5_count=$(printf '%s\n' "${text}" | grep -c 'q5_behavior_semantics\.py' || true)
  hyper_count=$(printf '%s\n' "${text}" | grep -c 'hyperopt_tune\.py' || true)
  other_count=$(printf '%s\n' "${text}" | grep -vc 'q2_routing_control\.py\|q3_stage_structure\.py\|q4_efficiency\.py\|q5_behavior_semantics\.py\|hyperopt_tune\.py' || true)

  [[ "${q2_count}" -gt 0 ]] && echo "  - q2_routing_control.py: ${q2_count}"
  [[ "${q3_count}" -gt 0 ]] && echo "  - q3_stage_structure.py: ${q3_count}"
  [[ "${q4_count}" -gt 0 ]] && echo "  - q4_efficiency.py: ${q4_count}"
  [[ "${q5_count}" -gt 0 ]] && echo "  - q5_behavior_semantics.py: ${q5_count}"
  [[ "${hyper_count}" -gt 0 ]] && echo "  - hyperopt_tune.py: ${hyper_count}"
  [[ "${other_count}" -gt 0 ]] && echo "  - other: ${other_count}"
  return 0
}

cd "${REPO_ROOT}"

echo "[appendix-wait] waiting for real_final_ablation main-body jobs to finish"
while true; do
  if running="$(main_ablation_jobs_running)"; then
    echo "[appendix-wait] still running at $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
    summarize_running_jobs "${running}"
    sleep "${POLL_SECONDS}"
    continue
  fi
  break
done

echo "[appendix-run] main-body jobs finished; starting KuaiRec appendix minimal sweep"
echo "[appendix-run] dataset=${DATASET} topk=${TOPK} seeds=${SEEDS} max_evals=${MAX_EVALS} gpus=${GPU_LIST}"
"${PY_BIN}" "${APPENDIX_DIR}/run_appendix_suite.py" \
  --datasets "${DATASET}" \
  --models featured_moe_n3 \
  --top-k-configs "${TOPK}" \
  --seeds "${SEEDS}" \
  --gpus "${GPU_LIST}" \
  --base-csv "${BASE_CSV}" \
  --max-evals "${MAX_EVALS}" \
  --max-run-hours "${MAX_RUN_HOURS}" \
  --tune-epochs "${TUNE_EPOCHS}" \
  --tune-patience "${TUNE_PATIENCE}" \
  --lr-mode "${LR_MODE}" \
  --search-algo "${SEARCH_ALGO}" \
  ${RESUME_FLAG} \
  ${DRY_RUN_FLAG} \
  --sections "${SECTIONS}"

echo "[appendix-run] complete"
