#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=/dev/null
source "${RUN_DIR}/common/dispatch_gpu_queue.sh"

DATASET="movielens1m"
GPU_LIST="0,1"
COMBOS_PER_GPU="2"
MAX_EVALS="8"
TUNE_EPOCHS="100"
TUNE_PATIENCE="12"
SEED_BASE="4500"
PHASE_PREFIX="P4V4D"
ROUTER_DESIGN="flat_legacy"
ROUTER_CLONE_RESIDUAL_SCALE="0.00"
TEACHER_DESIGN="group_comp_stat12"
TEACHER_DELIVERY="distill_and_fused_bias"
TEACHER_STAGE_MASK="mid_micro_only"
DRY_RUN="${DRY_RUN:-false}"
LOG_WANDB="false"

usage() {
  cat <<USAGE
Usage: $0 [--dataset movielens1m] [--gpus 0,1]
          [--router-design flat_legacy]
          [--teacher-design group_comp_stat12]
          [--teacher-delivery distill_and_fused_bias]
          [--teacher-stage-mask mid_micro_only]
          [--combos-per-gpu 2] [--max-evals 8]
          [--tune-epochs 100] [--tune-patience 12]
          [--phase-prefix P4V4D] [--dry-run]
USAGE
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2 ;;
    --gpus) GPU_LIST="$2"; shift 2 ;;
    --router-design) ROUTER_DESIGN="$2"; shift 2 ;;
    --router-clone-residual-scale) ROUTER_CLONE_RESIDUAL_SCALE="$2"; shift 2 ;;
    --teacher-design) TEACHER_DESIGN="$2"; shift 2 ;;
    --teacher-delivery) TEACHER_DELIVERY="$2"; shift 2 ;;
    --teacher-stage-mask) TEACHER_STAGE_MASK="$2"; shift 2 ;;
    --combos-per-gpu) COMBOS_PER_GPU="$2"; shift 2 ;;
    --max-evals) MAX_EVALS="$2"; shift 2 ;;
    --tune-epochs) TUNE_EPOCHS="$2"; shift 2 ;;
    --tune-patience) TUNE_PATIENCE="$2"; shift 2 ;;
    --phase-prefix) PHASE_PREFIX="$2"; shift 2 ;;
    --log-wandb) LOG_WANDB="true"; shift ;;
    --no-wandb) LOG_WANDB="false"; shift ;;
    --dry-run) DRY_RUN="true"; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

dispatch_parse_csv "$GPU_LIST" GPUS
[ "${#GPUS[@]}" -eq 0 ] && { echo "Empty GPU list" >&2; exit 1; }

build_catalog() {
  python3 - <<'PY'
rows = [
    ("C00", "0.0010", "0.10", "0.15", "3e-4,4.5e-4,6e-4,8e-4,1.1e-3,1.5e-3", "WEAK"),
    ("C01", "0.0020", "0.20", "0.25", "2.5e-4,3.5e-4,5e-4,7e-4,9e-4,1.2e-3", "MAIN"),
    ("C02", "0.0035", "0.35", "0.35", "1.5e-4,2.5e-4,3.5e-4,5e-4,7e-4,9e-4", "STRONG"),
]
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
COVERED=$(( ${#GPUS[@]} * COMBOS_PER_GPU ))
[ "${COVERED}" -gt "${TOTAL_COMBOS}" ] && COVERED="${TOTAL_COMBOS}"

run_one_combo() {
  local gpu="$1"
  local combo_idx="$2"
  local row
  row="$(read_combo "${COMBO_CATALOG}" "${combo_idx}")"
  IFS=$'\t' read -r combo_id kl_lambda bias_scale until lr_space tag <<< "${row}"
  local seed=$(( SEED_BASE + combo_idx ))
  local phase="${PHASE_PREFIX}_${combo_id}_${tag}"
  local cmd=(
    bash "${SCRIPT_DIR}/tune_hparam.sh"
    --dataset "${DATASET}"
    --gpu "${gpu}"
    --max-evals "${MAX_EVALS}"
    --tune-epochs "${TUNE_EPOCHS}"
    --tune-patience "${TUNE_PATIENCE}"
    --seed "${seed}"
    --phase "${phase}"
    --search-profile "bias"
    --lr-space "${lr_space}"
    --wd-space "5e-5"
    --dropout-space "0.10"
    --balance-space "0.005"
    --exp-name "fmoe_v4d_p4_strength_until"
    --exp-desc "v4_distillation P4: narrow teacher strength/until presets."
    --exp-focus "router_design,teacher_design,teacher_delivery,teacher_stage_mask,teacher_kl_lambda,teacher_bias_scale,teacher_until,learning_rate"
    --override "router_design=${ROUTER_DESIGN}"
    --override "++search.router_design=[${ROUTER_DESIGN}]"
    --override "router_clone_residual_scale=${ROUTER_CLONE_RESIDUAL_SCALE}"
    --override "++search.router_clone_residual_scale=[${ROUTER_CLONE_RESIDUAL_SCALE}]"
    --override "teacher_design=${TEACHER_DESIGN}"
    --override "++search.teacher_design=[${TEACHER_DESIGN}]"
    --override "teacher_delivery=${TEACHER_DELIVERY}"
    --override "++search.teacher_delivery=[${TEACHER_DELIVERY}]"
    --override "teacher_stage_mask=${TEACHER_STAGE_MASK}"
    --override "++search.teacher_stage_mask=[${TEACHER_STAGE_MASK}]"
    --override "teacher_kl_lambda=${kl_lambda}"
    --override "++search.teacher_kl_lambda=[${kl_lambda}]"
    --override "teacher_bias_scale=${bias_scale}"
    --override "++search.teacher_bias_scale=[${bias_scale}]"
    --override "teacher_temperature=1.5"
    --override "++search.teacher_temperature=[1.5]"
    --override "teacher_until=${until}"
    --override "++search.teacher_until=[${until}]"
    --override "teacher_stat_sharpness=16.0"
    --override "++search.teacher_stat_sharpness=[16.0]"
  )
  if [ "${LOG_WANDB}" = "true" ]; then
    cmd+=(--log-wandb)
  fi
  if [ "${DRY_RUN}" = "true" ]; then
    cmd+=(--dry-run)
  fi
  printf '[P4][%s][GPU %s] ' "${tag}" "${gpu}"
  printf '%q ' "${cmd[@]}"
  printf '\n'
  "${cmd[@]}"
}

trap 'dispatch_terminate_all GPUS; exit 130' INT TERM

for combo_idx in $(seq 0 $((COVERED - 1))); do
  dispatch_wait_for_gpu GPUS
  gpu="${FREE_GPU}"
  (
    set -euo pipefail
    run_one_combo "${gpu}" "${combo_idx}"
  ) &
  dispatch_set_pid "${gpu}" "$!"
done
dispatch_wait_all
