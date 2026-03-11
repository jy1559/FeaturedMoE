#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=/dev/null
source "${RUN_DIR}/common/dispatch_gpu_queue.sh"

DATASET="movielens1m"
GPU_LIST="0,1,2,3,4,5,6,7"
COMBOS_PER_GPU="2"
MAX_EVALS="20"
TUNE_EPOCHS="30"
TUNE_PATIENCE="5"
TRAIN_BATCH_SIZE="6144"
EVAL_BATCH_SIZE="16384"
SEED_BASE="4700"
PHASE_PREFIX="P2CMP16"
DRY_RUN="${DRY_RUN:-false}"
LOG_WANDB="false"

usage() {
  cat <<USAGE
Usage: $0 [--dataset movielens1m] [--gpus 0,1,2,3,4,5,6,7]
          [--combos-per-gpu 2] [--max-evals 20]
          [--tune-epochs 30] [--tune-patience 5]
          [--train-batch-size 6144] [--eval-batch-size 16384]
          [--phase-prefix P2CMP16] [--dry-run]
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
    --train-batch-size) TRAIN_BATCH_SIZE="$2"; shift 2 ;;
    --eval-batch-size) EVAL_BATCH_SIZE="$2"; shift 2 ;;
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
    ("C00", 7, "serial",   "learned",   "{}",                           "group_local_stat12", "distill_and_fused_bias", "mid_micro_only", "0.002", "0.20", "0.25", "ratio_bins", "5",  "1.8e-4,9.0e-4", "teacher", "BASE_DB"),
    ("C01", 7, "serial",   "learned",   "{}",                           "group_local_stat12", "fused_bias",            "mid_micro_only", "0.0",   "0.20", "0.25", "ratio_bins", "5",  "1.8e-4,9.0e-4", "teacher", "GLS_BIAS"),
    ("C02", 7, "serial",   "learned",   "{}",                           "rule_soft12",        "distill_and_fused_bias", "mid_micro_only", "0.002", "0.20", "0.25", "ratio_bins", "5",  "1.8e-4,9.0e-4", "teacher", "RULESOFT_TEACHER"),
    ("C03", 7, "serial",   "learned",   "{mid:rule_soft,micro:rule_soft}", "none",          "none",                  "all",            "0.0",   "0.0",  "0.25", "teacher_gls", "5", "2.5e-3,7.0e-3", "router",  "GLS_RULE_HYBRID"),
    ("C04", 7, "serial",   "learned",   "{}",                           "group_comp_stat12",  "distill_and_fused_bias", "mid_micro_only", "0.002", "0.20", "0.25", "ratio_bins", "5",  "1.8e-4,9.0e-4", "teacher", "GCS_SWAP"),
    ("C05", 7, "serial",   "learned",   "{}",                           "group_local_stat12", "distill_and_fused_bias", "mid_micro_only", "0.001", "0.10", "0.15", "ratio_bins", "5",  "1.8e-4,9.0e-4", "teacher", "WEAK"),
    ("C06", 7, "serial",   "learned",   "{}",                           "group_local_stat12", "distill_kl",            "mid_micro_only", "0.002", "0.0",  "0.25", "ratio_bins", "5",  "1.8e-4,9.0e-4", "teacher", "GLS_DKL"),
    ("C07", 7, "serial",   "learned",   "{}",                           "none",               "none",                  "all",            "0.0",   "0.0",  "0.25", "ratio_bins", "5",  "2.5e-3,7.0e-3", "router",  "PLAIN"),
    ("C08", 7, "serial",   "learned",   "{mid:rule_soft,micro:rule_soft}", "none",          "none",                  "all",            "0.0",   "0.0",  "0.25", "ratio_bins", "5", "2.5e-3,7.0e-3", "router",  "RULE_HYBRID_SOFT"),
    ("C09", 7, "serial",   "rule_soft", "{}",                           "none",               "none",                  "all",            "0.0",   "0.0",  "0.25", "ratio_bins", "5",  "4.0e-4,1.6e-3", "router",  "RULE_FULL_SOFT"),
    ("C10", 7, "serial",   "rule_soft", "{}",                           "none",               "none",                  "all",            "0.0",   "0.0",  "0.25", "ratio_bins", "10", "4.0e-4,1.6e-3", "router",  "RULE_FULL_SOFT_NB10"),
    ("C11", 7, "serial",   "learned",   "{}",                           "group_local_stat12", "distill_and_fused_bias", "all",            "0.002", "0.20", "0.25", "ratio_bins", "5",  "1.8e-4,9.0e-4", "teacher", "ALL_L7"),
    ("C12", 24, "parallel","learned",   "{}",                           "group_local_stat12", "distill_and_fused_bias", "all",            "0.002", "0.20", "0.25", "ratio_bins", "5",  "1.8e-4,9.0e-4", "teacher", "L24_MACRO_ONLY"),
    ("C13", 9, "serial",   "learned",   "{}",                           "group_local_stat12", "distill_and_fused_bias", "mid_micro_only", "0.002", "0.20", "0.25", "ratio_bins", "5",  "1.8e-4,9.0e-4", "teacher", "L9_MM"),
    ("C14", 16, "serial",  "learned",   "{}",                           "group_local_stat12", "distill_and_fused_bias", "mid_micro_only", "0.002", "0.20", "0.25", "ratio_bins", "5",  "1.8e-4,9.0e-4", "teacher", "L16_MM"),
    ("C15", 29, "parallel","learned",   "{}",                           "group_local_stat12", "distill_and_fused_bias", "mid_micro_only", "0.002", "0.20", "0.25", "ratio_bins", "5",  "1.8e-4,9.0e-4", "teacher", "L29_MM"),
]
print(";".join("\t".join(map(str, row)) for row in rows))
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
  IFS=$'\t' read -r combo_id layout_id execution router_impl router_impl_by_stage teacher_design teacher_delivery teacher_stage_mask teacher_kl_lambda teacher_bias_scale teacher_until rule_variant rule_n_bins lr_space search_profile tag <<< "${row}"
  local seed=$(( SEED_BASE + combo_idx ))
  local phase="${PHASE_PREFIX}_${combo_id}_${tag}"
  local cmd=(
    bash "${SCRIPT_DIR}/tune_hparam.sh"
    --dataset "${DATASET}"
    --layout-id "${layout_id}"
    --execution "${execution}"
    --gpu "${gpu}"
    --max-evals "${MAX_EVALS}"
    --tune-epochs "${TUNE_EPOCHS}"
    --tune-patience "${TUNE_PATIENCE}"
    --train-batch-size "${TRAIN_BATCH_SIZE}"
    --eval-batch-size "${EVAL_BATCH_SIZE}"
    --seed "${seed}"
    --phase "${phase}"
    --search-profile "${search_profile}"
    --lr-space "${lr_space}"
    --wd-space "5e-5"
    --dropout-space "0.10"
    --balance-space "0.005"
    --exp-name "fmoe_v4d_p2_compare16"
    --exp-desc "v4_distillation Phase 2 compare-16: GLS distill vs rule_soft teacher vs direct rule-hybrid and layout controls on flat_legacy."
    --exp-focus "fmoe_v2_layout_id,fmoe_stage_execution_mode,router_impl,router_impl_by_stage,rule_router.variant,rule_router.n_bins,teacher_design,teacher_delivery,teacher_stage_mask,teacher_kl_lambda,teacher_bias_scale,teacher_until,learning_rate"
    --override "router_design=flat_legacy"
    --override "++search.router_design=[flat_legacy]"
    --override "router_clone_residual_scale=0.20"
    --override "++search.router_clone_residual_scale=[0.20]"
    --override "router_impl=${router_impl}"
    --override "++search.router_impl=[${router_impl}]"
    --override "++router_impl_by_stage=${router_impl_by_stage}"
    --override "++search.router_impl_by_stage=[${router_impl_by_stage}]"
    --override "rule_router.variant=${rule_variant}"
    --override "++search={rule_router.variant:[${rule_variant}]}"
    --override "rule_router.n_bins=${rule_n_bins}"
    --override "++search={rule_router.n_bins:[${rule_n_bins}]}"
    --override "rule_router.feature_per_expert=4"
    --override "++search={rule_router.feature_per_expert:[4]}"
    --override "teacher_design=${teacher_design}"
    --override "++search.teacher_design=[${teacher_design}]"
    --override "teacher_delivery=${teacher_delivery}"
    --override "++search.teacher_delivery=[${teacher_delivery}]"
    --override "teacher_stage_mask=${teacher_stage_mask}"
    --override "++search.teacher_stage_mask=[${teacher_stage_mask}]"
    --override "teacher_kl_lambda=${teacher_kl_lambda}"
    --override "++search.teacher_kl_lambda=[${teacher_kl_lambda}]"
    --override "teacher_bias_scale=${teacher_bias_scale}"
    --override "++search.teacher_bias_scale=[${teacher_bias_scale}]"
    --override "teacher_temperature=1.5"
    --override "++search.teacher_temperature=[1.5]"
    --override "teacher_until=${teacher_until}"
    --override "++search.teacher_until=[${teacher_until}]"
    --override "teacher_stat_sharpness=16.0"
    --override "++search.teacher_stat_sharpness=[16.0]"
    --override "search_space_type_overrides.learning_rate=loguniform"
  )
  if [ "${LOG_WANDB}" = "true" ]; then
    cmd+=(--log-wandb)
  fi
  if [ "${DRY_RUN}" = "true" ]; then
    cmd+=(--dry-run)
  fi
  printf '[CMP16][%s][GPU %s] ' "${tag}" "${gpu}"
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
