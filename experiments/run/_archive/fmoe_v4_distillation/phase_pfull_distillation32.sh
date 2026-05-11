#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=/dev/null
source "${RUN_DIR}/common/dispatch_gpu_queue.sh"

DATASET="movielens1m"
GPU_LIST="0,1,2,3,4,5,6,7"
COMBOS_PER_GPU="4"
MAX_EVALS="10"
TUNE_EPOCHS="100"
TUNE_PATIENCE="10"
TRAIN_BATCH_SIZE="6144"
EVAL_BATCH_SIZE="16384"
SEED_BASE="4600"
PHASE_PREFIX="PFULLV4D"
DRY_RUN="${DRY_RUN:-false}"
LOG_WANDB="false"

usage() {
  cat <<USAGE
Usage: $0 [--dataset movielens1m] [--gpus 0,1,2,3,4,5,6,7]
          [--combos-per-gpu 4] [--max-evals 10]
          [--tune-epochs 100] [--tune-patience 10]
          [--train-batch-size 6144] [--eval-batch-size 16384]
          [--phase-prefix PFULLV4D] [--dry-run]
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
teachers = [
    ("group_local_stat12", "GLS"),
    ("group_comp_stat12", "GCS"),
    ("group_comp_shape12", "GSH"),
]

controls = [
    ("C00", "none", "none", "all", "0.0", "0.0", "0.25", "8e-4,6.0e-3", "{}", "PLAIN"),
    ("C01", "none", "none", "all", "0.0", "0.0", "0.25", "8e-4,6.0e-3", "{mid:rule_soft,micro:rule_soft}", "LEGACY_HYBRID"),
]

teacher_profiles = {
    "GLS": [
        ("distill_kl", "mid_micro_only", "main",   "0.002",  "0.0",  "0.25", "2.5e-4,2.4e-3", "DKL_MM_MAIN"),
        ("fused_bias", "mid_micro_only", "main",   "0.0",    "0.20", "0.25", "2.5e-4,2.1e-3", "BIAS_MM_MAIN"),
        ("distill_and_fused_bias", "mid_micro_only", "weak",   "0.001",  "0.10", "0.15", "2.2e-4,1.8e-3", "DB_MM_WEAK"),
        ("distill_and_fused_bias", "mid_micro_only", "main",   "0.002",  "0.20", "0.25", "1.8e-4,1.5e-3", "DB_MM_MAIN"),
        ("distill_and_fused_bias", "mid_micro_only", "strong", "0.0035", "0.35", "0.35", "1.2e-4,1.0e-3", "DB_MM_STRONG"),
        ("distill_and_fused_bias", "all",            "main",   "0.002",  "0.20", "0.25", "1.8e-4,1.5e-3", "DB_ALL_MAIN"),
    ],
    "GCS": [
        ("distill_kl", "mid_micro_only", "main",   "0.002",  "0.0",  "0.25", "2.5e-4,2.6e-3", "DKL_MM_MAIN"),
        ("distill_kl", "all",            "main",   "0.002",  "0.0",  "0.25", "2.0e-4,2.4e-3", "DKL_ALL_MAIN"),
        ("fused_bias", "mid_micro_only", "weak",   "0.0",    "0.10", "0.15", "2.5e-4,2.1e-3", "BIAS_MM_WEAK"),
        ("fused_bias", "mid_micro_only", "main",   "0.0",    "0.20", "0.25", "2.2e-4,1.9e-3", "BIAS_MM_MAIN"),
        ("fused_bias", "mid_micro_only", "strong", "0.0",    "0.35", "0.35", "1.5e-4,1.3e-3", "BIAS_MM_STRONG"),
        ("fused_bias", "all",            "main",   "0.0",    "0.20", "0.25", "2.2e-4,2.1e-3", "BIAS_ALL_MAIN"),
        ("distill_and_fused_bias", "mid_micro_only", "weak",   "0.001",  "0.10", "0.15", "2.0e-4,1.7e-3", "DB_MM_WEAK"),
        ("distill_and_fused_bias", "mid_micro_only", "main",   "0.002",  "0.20", "0.25", "1.5e-4,1.3e-3", "DB_MM_MAIN"),
        ("distill_and_fused_bias", "mid_micro_only", "strong", "0.0035", "0.35", "0.35", "1.0e-4,1.0e-3",  "DB_MM_STRONG"),
        ("distill_and_fused_bias", "all",            "weak",   "0.001",  "0.10", "0.15", "2.0e-4,1.8e-3", "DB_ALL_WEAK"),
        ("distill_and_fused_bias", "all",            "main",   "0.002",  "0.20", "0.25", "1.5e-4,1.5e-3", "DB_ALL_MAIN"),
        ("distill_and_fused_bias", "all",            "strong", "0.0035", "0.35", "0.35", "1.0e-4,1.1e-3",  "DB_ALL_STRONG"),
    ],
    "GSH": [
        ("distill_kl", "mid_micro_only", "main",   "0.002",  "0.0",  "0.25", "2.5e-4,2.6e-3", "DKL_MM_MAIN"),
        ("distill_kl", "all",            "main",   "0.002",  "0.0",  "0.25", "2.0e-4,2.4e-3", "DKL_ALL_MAIN"),
        ("fused_bias", "mid_micro_only", "weak",   "0.0",    "0.10", "0.15", "2.5e-4,2.1e-3", "BIAS_MM_WEAK"),
        ("fused_bias", "mid_micro_only", "main",   "0.0",    "0.20", "0.25", "2.2e-4,1.9e-3", "BIAS_MM_MAIN"),
        ("fused_bias", "mid_micro_only", "strong", "0.0",    "0.35", "0.35", "1.5e-4,1.3e-3", "BIAS_MM_STRONG"),
        ("fused_bias", "all",            "main",   "0.0",    "0.20", "0.25", "2.2e-4,2.1e-3", "BIAS_ALL_MAIN"),
        ("distill_and_fused_bias", "mid_micro_only", "weak",   "0.001",  "0.10", "0.15", "2.0e-4,1.7e-3", "DB_MM_WEAK"),
        ("distill_and_fused_bias", "mid_micro_only", "main",   "0.002",  "0.20", "0.25", "1.5e-4,1.3e-3", "DB_MM_MAIN"),
        ("distill_and_fused_bias", "mid_micro_only", "strong", "0.0035", "0.35", "0.35", "1.0e-4,1.0e-3",  "DB_MM_STRONG"),
        ("distill_and_fused_bias", "all",            "weak",   "0.001",  "0.10", "0.15", "2.0e-4,1.8e-3", "DB_ALL_WEAK"),
        ("distill_and_fused_bias", "all",            "main",   "0.002",  "0.20", "0.25", "1.5e-4,1.5e-3", "DB_ALL_MAIN"),
        ("distill_and_fused_bias", "all",            "strong", "0.0035", "0.35", "0.35", "1.0e-4,1.1e-3",  "DB_ALL_STRONG"),
    ],
}

rows = list(controls)
combo_idx = len(rows)
for teacher_design, teacher_tag in teachers:
    for delivery, stage_mask, strength_tag, kl_lambda, bias_scale, until, lr_space, tag in teacher_profiles[teacher_tag]:
        rows.append(
            (
                f"C{combo_idx:02d}",
                teacher_design,
                delivery,
                stage_mask,
                kl_lambda,
                bias_scale,
                until,
                lr_space,
                "{}",
                f"{teacher_tag}_{tag}",
            )
        )
        combo_idx += 1

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
  IFS=$'\t' read -r combo_id teacher_design teacher_delivery teacher_stage_mask teacher_kl_lambda teacher_bias_scale teacher_until lr_space router_impl_by_stage tag <<< "${row}"
  local seed=$(( SEED_BASE + combo_idx ))
  local phase="${PHASE_PREFIX}_${combo_id}_${tag}"
  local cmd=(
    bash "${SCRIPT_DIR}/tune_hparam.sh"
    --dataset "${DATASET}"
    --gpu "${gpu}"
    --max-evals "${MAX_EVALS}"
    --tune-epochs "${TUNE_EPOCHS}"
    --tune-patience "${TUNE_PATIENCE}"
    --train-batch-size "${TRAIN_BATCH_SIZE}"
    --eval-batch-size "${EVAL_BATCH_SIZE}"
    --seed "${seed}"
    --phase "${phase}"
    --search-profile "teacher"
    --lr-space "${lr_space}"
    --wd-space "5e-5"
    --dropout-space "0.10"
    --balance-space "0.005"
    --exp-name "fmoe_v4d_full_distill32"
    --exp-desc "v4_distillation weighted 32-combo distillation sweep emphasizing group_comp teachers, fused bias, and mid/micro-only delivery with larger ML1 batch."
    --exp-focus "teacher_design,teacher_delivery,teacher_stage_mask,teacher_kl_lambda,teacher_bias_scale,teacher_until,router_impl_by_stage,learning_rate"
    --override "router_design=flat_legacy"
    --override "++search.router_design=[flat_legacy]"
    --override "router_clone_residual_scale=0.20"
    --override "++search.router_clone_residual_scale=[0.20]"
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
    --override "++router_impl_by_stage=${router_impl_by_stage}"
    --override "++search.router_impl_by_stage=[${router_impl_by_stage}]"
  )
  if [ "${LOG_WANDB}" = "true" ]; then
    cmd+=(--log-wandb)
  fi
  if [ "${DRY_RUN}" = "true" ]; then
    cmd+=(--dry-run)
  fi
  printf '[FULL32][%s][GPU %s] ' "${tag}" "${gpu}"
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
