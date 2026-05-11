#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=/dev/null
source "${RUN_DIR}/common/dispatch_gpu_queue.sh"

DATASET="movielens1m"
GPU_LIST="0,1"
COMBOS_PER_GPU="3"
MAX_EVALS="8"
TUNE_EPOCHS="100"
TUNE_PATIENCE="12"
SEED_BASE="4200"
PHASE_PREFIX="P1V4D"
DRY_RUN="${DRY_RUN:-false}"
LOG_WANDB="false"

usage() {
  cat <<USAGE
Usage: $0 [--dataset movielens1m] [--gpus 0,1]
          [--combos-per-gpu 3] [--max-evals 8]
          [--tune-epochs 100] [--tune-patience 12]
          [--phase-prefix P1V4D] [--dry-run]
USAGE
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2 ;;
    --gpus) GPU_LIST="$2"; shift 2 ;;
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
    ("C00", "none",                "none",       "all",            "{}",                           "6e-4,9e-4,1.2e-3,1.6e-3,2.1e-3,2.8e-3,3.6e-3,4.8e-3", "NONE"),
    ("C01", "group_local_stat12",  "distill_kl", "all",            "{}",                           "2e-4,3e-4,4.5e-4,6e-4,8e-4,1.1e-3,1.5e-3,2.0e-3",     "GLS12"),
    ("C02", "group_comp_stat12",   "distill_kl", "all",            "{}",                           "2e-4,3e-4,4.5e-4,6e-4,8e-4,1.1e-3,1.5e-3,2.0e-3",     "GCS12"),
    ("C03", "group_comp_shape12",  "distill_kl", "all",            "{}",                           "1.5e-4,2.5e-4,3.5e-4,5e-4,7e-4,9e-4,1.2e-3,1.6e-3",   "GSH12"),
    ("C04", "none",                "none",       "all",            "{mid:rule_soft,micro:rule_soft}", "6e-4,9e-4,1.2e-3,1.6e-3,2.1e-3,2.8e-3,3.6e-3,4.8e-3", "LEGACY_HYBRID"),
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
  IFS=$'\t' read -r combo_id teacher_design teacher_delivery teacher_mask router_impl_by_stage lr_space tag <<< "${row}"
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
    --search-profile "teacher"
    --lr-space "${lr_space}"
    --wd-space "5e-5"
    --dropout-space "0.10"
    --balance-space "0.005"
    --exp-name "fmoe_v4d_p1_teacher_family"
    --exp-desc "v4_distillation P1: direct 12-way teacher family screen on flat_legacy."
    --exp-focus "teacher_design,teacher_delivery,teacher_stage_mask,teacher_kl_lambda,teacher_temperature,teacher_until,router_impl_by_stage,learning_rate"
    --override "router_design=flat_legacy"
    --override "++search.router_design=[flat_legacy]"
    --override "teacher_design=${teacher_design}"
    --override "++search.teacher_design=[${teacher_design}]"
    --override "teacher_delivery=${teacher_delivery}"
    --override "++search.teacher_delivery=[${teacher_delivery}]"
    --override "teacher_stage_mask=${teacher_mask}"
    --override "++search.teacher_stage_mask=[${teacher_mask}]"
    --override "teacher_kl_lambda=0.002"
    --override "++search.teacher_kl_lambda=[0.002]"
    --override "teacher_bias_scale=0.0"
    --override "++search.teacher_bias_scale=[0.0]"
    --override "teacher_temperature=1.5"
    --override "++search.teacher_temperature=[1.5]"
    --override "teacher_until=0.25"
    --override "++search.teacher_until=[0.25]"
    --override "teacher_stat_sharpness=16.0"
    --override "++search.teacher_stat_sharpness=[16.0]"
    --override "++router_impl_by_stage=${router_impl_by_stage}"
    --override "++search.router_impl_by_stage=[${router_impl_by_stage}]"
  )
  if [ "${tag}" = "LEGACY_HYBRID" ]; then
    cmd+=(
      --override "teacher_design=none"
      --override "++search.teacher_design=[none]"
      --override "teacher_delivery=none"
      --override "++search.teacher_delivery=[none]"
      --override "teacher_kl_lambda=0.0"
      --override "++search.teacher_kl_lambda=[0.0]"
    )
  fi
  if [ "${LOG_WANDB}" = "true" ]; then
    cmd+=(--log-wandb)
  fi
  if [ "${DRY_RUN}" = "true" ]; then
    cmd+=(--dry-run)
  fi
  printf '[P1][%s][GPU %s] ' "${tag}" "${gpu}"
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
