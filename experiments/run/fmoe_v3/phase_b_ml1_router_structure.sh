#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=/dev/null
source "${RUN_DIR}/common/dispatch_gpu_queue.sh"

DATASETS="movielens1m"
GPU_LIST="0,1"
COMBOS_PER_GPU="3"
MAX_EVALS="6"
TUNE_EPOCHS="60"
TUNE_PATIENCE="6"
SEED_BASE="2410"
PHASE_PREFIX="V3B"
CATALOG_PROFILE="quick6"
LOG_WANDB="false"
DRY_RUN="${DRY_RUN:-false}"

usage() {
  cat <<USAGE
Usage: $0 [--datasets movielens1m] [--gpus 0,1,2,3]
          [--catalog-profile quick6]
          [--combos-per-gpu 3] [--max-evals 6]
          [--tune-epochs 60] [--tune-patience 6]
          [--phase-prefix V3B] [--seed-base 2410]
          [--log-wandb] [--dry-run]
USAGE
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --datasets) DATASETS="$2"; shift 2 ;;
    --gpus) GPU_LIST="$2"; shift 2 ;;
    --catalog-profile) CATALOG_PROFILE="$2"; shift 2 ;;
    --combos-per-gpu) COMBOS_PER_GPU="$2"; shift 2 ;;
    --max-evals) MAX_EVALS="$2"; shift 2 ;;
    --tune-epochs) TUNE_EPOCHS="$2"; shift 2 ;;
    --tune-patience) TUNE_PATIENCE="$2"; shift 2 ;;
    --phase-prefix) PHASE_PREFIX="$2"; shift 2 ;;
    --seed-base) SEED_BASE="$2"; shift 2 ;;
    --log-wandb) LOG_WANDB="true"; shift ;;
    --no-wandb) LOG_WANDB="false"; shift ;;
    --dry-run) DRY_RUN="true"; shift ;;
    --help|-h) usage; exit 0 ;;
    --) shift; break ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

dispatch_parse_csv "$GPU_LIST" GPUS
[ "${#GPUS[@]}" -eq 0 ] && { echo "Empty GPU list" >&2; exit 1; }
dispatch_parse_csv "$DATASETS" DATASET_ARR
[ "${#DATASET_ARR[@]}" -eq 0 ] && { echo "Empty dataset list" >&2; exit 1; }

build_catalog() {
  python3 - <<'PY' "$CATALOG_PROFILE"
import sys
profile = sys.argv[1]
if profile == "quick6":
    rows = [
        ("C00", "flat_legacy",              "true",  "0.5", "0.5", "1e-4,4e-4,7e-4,0.0011,0.0016,0.0024,0.0034,0.0048,0.0062,0.0080", "LEGACY"),
        ("C01", "flat_hidden_only",         "false", "0.5", "0.5", "1e-4,3e-4,5e-4,8e-4,0.0012,0.0018,0.0026,0.0038,0.0055,0.0080", "HONLY"),
        ("C02", "flat_global_interaction",  "true",  "0.5", "0.5", "1e-4,3e-4,5e-4,8e-4,0.0012,0.0018,0.0026,0.0038,0.0055,0.0080", "GINT"),
        ("C03", "flat_hidden_group_clone12","true",  "0.5", "0.5", "1e-4,2e-4,3.5e-4,5e-4,7e-4,0.0010,0.0014,0.0020,0.0035,0.0080", "HGCLONE"),
        ("C04", "flat_clone_residual12",    "true",  "0.5", "0.5", "1e-4,2e-4,3.5e-4,5e-4,7e-4,0.0010,0.0014,0.0020,0.0035,0.0080", "CRES"),
        ("C05", "flat_group_clone_combo",   "true",  "0.5", "0.5", "1e-4,2e-4,3e-4,4.5e-4,6.5e-4,9e-4,0.0013,0.0019,0.0032,0.0080", "GCOMBO"),
    ]
else:
    raise SystemExit(f"Unsupported --catalog-profile={profile}")
print(";".join("\t".join(row) for row in rows))
PY
}

read_combo() {
  local raw="$1"
  local idx="$2"
  python3 - <<'PY' "$raw" "$idx"
import sys
rows = [r for r in sys.argv[1].split(";") if r.strip()]
print(rows[int(sys.argv[2])])
PY
}

combo_count() {
  local raw="$1"
  python3 - <<'PY' "$raw"
import sys
rows = [r for r in sys.argv[1].split(";") if r.strip()]
print(len(rows))
PY
}

COMBO_CATALOG="$(build_catalog)"
TOTAL_COMBOS="$(combo_count "${COMBO_CATALOG}")"
if [ -z "${COMBOS_PER_GPU}" ]; then
  COMBOS_PER_GPU=$(( (TOTAL_COMBOS + ${#GPUS[@]} - 1) / ${#GPUS[@]} ))
fi

covered=$(( ${#GPUS[@]} * COMBOS_PER_GPU ))
if [ "${covered}" -gt "${TOTAL_COMBOS}" ]; then
  covered="${TOTAL_COMBOS}"
fi

WORKER_PIDS=()
on_interrupt() {
  echo "[INTERRUPT] stopping phase B workers..."
  dispatch_terminate_all GPUS
  exit 130
}
trap on_interrupt INT TERM

run_one_combo() {
  local dataset="$1"
  local gpu="$2"
  local combo_idx="$3"
  local row
  row="$(read_combo "${COMBO_CATALOG}" "${combo_idx}")"
  IFS=$'\t' read -r combo_id router_design use_feature bias_scale clone_scale lr_space tag <<< "${row}"
  local seed=$(( SEED_BASE + combo_idx ))
  local phase="${PHASE_PREFIX}_${combo_id}_${tag}"
  local cmd=(
    bash "${SCRIPT_DIR}/tune_hparam.sh"
    --dataset "${dataset}"
    --layout-id "7"
    --execution "serial"
    --gpu "${gpu}"
    --max-evals "${MAX_EVALS}"
    --tune-epochs "${TUNE_EPOCHS}"
    --tune-patience "${TUNE_PATIENCE}"
    --seed "${seed}"
    --phase "${phase}"
    --search-profile "wide"
    --train-batch-size "3072"
    --eval-batch-size "6144"
    --embedding-size "128"
    --d-feat-emb "16"
    --d-expert-hidden "512"
    --d-router-hidden "64"
    --expert-scale "3"
    --lr-space "${lr_space}"
    --wd-space "1e-6"
    --dropout-space "0.10"
    --balance-space "0.005"
    --exp-name "fmoe_v3_phase_b_router_structure"
    --exp-desc "ML1 flat-router structure screen over hidden/global/group-aware router variants."
    --exp-focus "router_design,router_use_feature,router_group_bias_scale,router_clone_residual_scale,moe_top_k,learning_rate,weight_decay,hidden_dropout_prob,balance_loss_lambda"
    --override "router_design=${router_design}"
    --override "++search.router_design=[${router_design}]"
    --override "weight_decay=1e-6"
    --override "router_use_hidden=true"
    --override "router_use_feature=${use_feature}"
    --override "router_impl=learned"
    --override "router_group_bias_scale=${bias_scale}"
    --override "++search.router_group_bias_scale=[${bias_scale}]"
    --override "router_clone_residual_scale=${clone_scale}"
    --override "++search.router_clone_residual_scale=[${clone_scale}]"
    --override "router_distill_enable=false"
    --override "++search.router_distill_enable=[false]"
    --override "router_distill_mode=none"
    --override "++search.router_distill_mode=[none]"
    --override "++router_impl_by_stage={}"
    --override "++search.router_impl_by_stage=[{}]"
    --override "moe_top_k=0"
    --override "++search.moe_top_k=[0]"
  )
  if [ "${LOG_WANDB}" = "true" ]; then
    cmd+=(--log-wandb)
  fi
  if [ "${DRY_RUN}" = "true" ]; then
    cmd+=(--dry-run)
  fi
  printf '[Phase B][%s][GPU %s] ' "${combo_id}" "${gpu}"
  printf '%q ' "${cmd[@]}"
  printf '\n'
  "${cmd[@]}"
}

for dataset in "${DATASET_ARR[@]}"; do
  echo "=== [${dataset}] Phase B router-structure (${covered} combos) ==="
  WORKER_PIDS=()
  for combo_idx in $(seq 0 $((covered - 1))); do
    dispatch_wait_for_gpu GPUS
    gpu="${FREE_GPU}"
    (
      set -euo pipefail
      run_one_combo "${dataset}" "${gpu}" "${combo_idx}"
    ) &
    pid=$!
    WORKER_PIDS+=("${pid}")
    dispatch_set_pid "${gpu}" "${pid}"
  done
  dispatch_wait_all
done
