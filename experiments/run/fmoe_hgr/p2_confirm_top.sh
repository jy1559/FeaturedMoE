#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=/dev/null
source "${RUN_DIR}/common/run_metadata.sh"
# shellcheck source=/dev/null
source "${RUN_DIR}/common/dispatch_gpu_queue.sh"

DATASETS="movielens1m"
GPU_LIST="0,1"
TOPN="4"
SOURCE_PHASE_PREFIX="P1HGR"
PHASE_PREFIX="P2HGR"
MAX_EVALS="20"
TUNE_EPOCHS="100"
TUNE_PATIENCE="10"
SEED_BASE="760"
SEARCH_PROFILE="confirm_narrow"
LOG_WANDB="false"
DRY_RUN="${DRY_RUN:-false}"
LR_SPACE="2.5e-4,5e-4,7.5e-4,1e-3,1.5e-3,2e-3,3e-3"
WD_SPACE="0,5e-6,1e-5,5e-5"
DROP_SPACE="0.08,0.12,0.16"
BAL_SPACE="0.003,0.01,0.03"
EXP_NAME="P2_hgr_confirm"
EXP_DESC="Re-run top P1 HGR combos with narrower search space to verify signal."
EXP_FOCUS="stage_merge_mode,group_router_mode,arch_layout_id,group_top_k,expert_use_feature,macro_routing_scope,parallel_stage_gate_temperature,learning_rate,weight_decay,hidden_dropout_prob,balance_loss_lambda"

usage() {
  cat <<USAGE
Usage: $0 [--datasets movielens1m] [--gpus 0,1] [--topn 4]
          [--source-phase-prefix P1HGR] [--phase-prefix P2HGR]
          [--max-evals 20] [--search-profile confirm_narrow]
USAGE
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --datasets) DATASETS="$2"; shift 2 ;;
    --gpus) GPU_LIST="$2"; shift 2 ;;
    --topn) TOPN="$2"; shift 2 ;;
    --source-phase-prefix) SOURCE_PHASE_PREFIX="$2"; shift 2 ;;
    --phase-prefix) PHASE_PREFIX="$2"; shift 2 ;;
    --max-evals) MAX_EVALS="$2"; shift 2 ;;
    --tune-epochs) TUNE_EPOCHS="$2"; shift 2 ;;
    --tune-patience) TUNE_PATIENCE="$2"; shift 2 ;;
    --seed-base) SEED_BASE="$2"; shift 2 ;;
    --search-profile) SEARCH_PROFILE="$2"; shift 2 ;;
    --lr-space) LR_SPACE="$2"; shift 2 ;;
    --wd-space) WD_SPACE="$2"; shift 2 ;;
    --dropout-space) DROP_SPACE="$2"; shift 2 ;;
    --balance-space) BAL_SPACE="$2"; shift 2 ;;
    --log-wandb) LOG_WANDB="true"; shift ;;
    --no-wandb) LOG_WANDB="false"; shift ;;
    --dry-run) DRY_RUN="true"; shift ;;
    --help|-h) usage; exit 0 ;;
    --) shift; break ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

dispatch_parse_csv "$GPU_LIST" GPUS
[ "${#GPUS[@]}" -eq 0 ] && { echo "Empty GPU list" >&2; exit 1; }
dispatch_parse_csv "$DATASETS" DS_ARR
[ "${#DS_ARR[@]}" -eq 0 ] && { echo "Empty dataset list" >&2; exit 1; }

EXP_DIR="$(run_experiments_dir)"
cd "$EXP_DIR"
run_export_runtime_env
PY_BIN="$(run_python_bin)"

phase_top_results() {
  local dataset="$1"
  local phase_prefix="$2"
  local topn="$3"
  "$PY_BIN" - <<'PY' "$dataset" "$phase_prefix" "$topn"
import json
import os
import sys
from pathlib import Path

dataset = sys.argv[1]
phase_prefix = sys.argv[2]
topn = int(sys.argv[3])
root = Path(os.environ.get("HYPEROPT_RESULTS_DIR", "run/artifacts/results")) / "fmoe_hgr"
if not root.exists():
    raise SystemExit(0)

rows = []
for p in root.glob("*.json"):
    try:
        d = json.load(open(p, "r", encoding="utf-8"))
    except Exception:
        continue
    if str(d.get("dataset", "")) != dataset:
        continue
    if "hgr" not in str(d.get("model", "")).lower():
        continue
    phase = str(d.get("run_phase", ""))
    if not phase.startswith(phase_prefix):
        continue
    score = d.get("best_mrr@20")
    if not isinstance(score, (int, float)):
        bvr = d.get("best_valid_result", {})
        score = bvr.get("mrr@20", float("-inf")) if isinstance(bvr, dict) else float("-inf")
    rows.append((float(score), p.stat().st_mtime, p))

rows.sort(key=lambda x: (x[0], x[1]), reverse=True)
for _, _, path in rows[:topn]:
    print(path)
PY
}

on_interrupt() {
  echo "[INTERRUPT] stopping HGR P2 workers..."
  dispatch_terminate_all GPUS || true
  exit 130
}
trap on_interrupt INT TERM

job_idx=0
for ds in "${DS_ARR[@]}"; do
  mapfile -t RESULTS < <(phase_top_results "$ds" "$SOURCE_PHASE_PREFIX" "$TOPN")
  if [ "${#RESULTS[@]}" -eq 0 ]; then
    echo "[P2-HGR] no results found for dataset=${ds} phase_prefix=${SOURCE_PHASE_PREFIX}" >&2
    continue
  fi

  echo "=== [${ds}] HGR P2 confirm (${#RESULTS[@]} reruns from ${SOURCE_PHASE_PREFIX}) ==="
  rank=0
  for result_json in "${RESULTS[@]}"; do
    rank=$((rank + 1))
    dispatch_wait_for_gpu GPUS
    gpu="$FREE_GPU"
    seed=$(( SEED_BASE + job_idx ))
    phase="${PHASE_PREFIX}_R$(printf '%02d' "$rank")"
    job_idx=$((job_idx + 1))

    cmd=(
      bash "${SCRIPT_DIR}/tune_hparam.sh"
      --dataset "${ds}"
      --gpu "${gpu}"
      --max-evals "${MAX_EVALS}"
      --tune-epochs "${TUNE_EPOCHS}"
      --tune-patience "${TUNE_PATIENCE}"
      --seed "${seed}"
      --phase "${phase}"
      --search-profile "${SEARCH_PROFILE}"
      --parent-result "${result_json}"
      --lr-space "${LR_SPACE}"
      --wd-space "${WD_SPACE}"
      --dropout-space "${DROP_SPACE}"
      --balance-space "${BAL_SPACE}"
      --exp-name "${EXP_NAME}"
      --exp-desc "${EXP_DESC} parent=$(basename "${result_json}")"
      --exp-focus "${EXP_FOCUS}"
    )

    if [ "$LOG_WANDB" = "true" ]; then
      cmd+=(--log-wandb)
    else
      cmd+=(--no-wandb)
    fi
    if [ "$DRY_RUN" = "true" ]; then
      cmd+=(--dry-run)
    fi

    echo "[P2-HGR] gpu=${gpu} rank=${rank} parent=$(basename "${result_json}")"
    setsid "${cmd[@]}" &
    pid=$!
    dispatch_set_pid "$gpu" "$pid"
  done
done

dispatch_wait_all
run_update_track_report fmoe_hgr
