#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=/dev/null
source "${RUN_DIR}/common/run_metadata.sh"
# shellcheck source=/dev/null
source "${RUN_DIR}/common/dispatch_gpu_queue.sh"

DATASETS="movielens1m"
GPU_LIST="0,1,2,3"
COMBOS_PER_GPU="1"
MAX_EVALS="10"
TUNE_EPOCHS="50"
TUNE_PATIENCE="10"
SEED_BASE="2610"
PHASE_PREFIX="R2OHGRv3_teacher4"
LOG_WANDB="false"
DRY_RUN="${DRY_RUN:-false}"
EXP_NAME_BASE="R2_hgr_v3_outer_teacher"
EXP_DESC_BASE="HGRv3 R2 outer/teacher ablation: all balance off, outer feature on/off, inner teacher strength comparison."
EXP_FOCUS="arch_layout_id,embedding_size,d_expert_hidden,d_router_hidden,expert_scale,outer_router_use_feature,inner_rule_mode,inner_rule_lambda,inner_rule_until,learning_rate"

usage() {
  cat <<USAGE
Usage: $0 [--datasets movielens1m] [--gpus 0,1,2,3]
          [--combos-per-gpu 1] [--max-evals 10] [--tune-epochs 50] [--tune-patience 10]
USAGE
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --datasets) DATASETS="$2"; shift 2 ;;
    --gpus) GPU_LIST="$2"; shift 2 ;;
    --combos-per-gpu) COMBOS_PER_GPU="$2"; shift 2 ;;
    --max-evals) MAX_EVALS="$2"; shift 2 ;;
    --tune-epochs) TUNE_EPOCHS="$2"; shift 2 ;;
    --tune-patience) TUNE_PATIENCE="$2"; shift 2 ;;
    --seed-base) SEED_BASE="$2"; shift 2 ;;
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
dispatch_parse_csv "$DATASETS" DATASET_ARR
[ "${#DATASET_ARR[@]}" -eq 0 ] && { echo "Empty dataset list" >&2; exit 1; }

generate_combo_rows() {
  python3 - <<'PY'
combos = [
    {
        "tag": "A1_feat_off",
        "layout": "15",
        "emb": "160",
        "dfeat": "16",
        "dexp": "256",
        "drouter": "112",
        "scale": "4",
        "train_bs": "2048",
        "eval_bs": "4096",
        "outer_feature": "true",
        "inner_mode": "off",
        "inner_lambda": "0.0",
        "inner_tau": "1.5",
        "inner_until": "0.2",
        "inner_bias": "0.0",
        "expert_top_k": "1",
        "base_lr": "5e-4",
        "lr_space": "1e-4,1.8e-4,2.5e-4,3.5e-4,5.0e-4,7.0e-4,1.0e-3,1.5e-3,2.5e-3,8e-3",
    },
    {
        "tag": "A1_feat_weak",
        "layout": "15",
        "emb": "160",
        "dfeat": "16",
        "dexp": "256",
        "drouter": "112",
        "scale": "4",
        "train_bs": "2048",
        "eval_bs": "4096",
        "outer_feature": "true",
        "inner_mode": "distill",
        "inner_lambda": "2e-3",
        "inner_tau": "1.5",
        "inner_until": "0.2",
        "inner_bias": "0.0",
        "expert_top_k": "1",
        "base_lr": "4e-4",
        "lr_space": "1e-4,1.5e-4,2.2e-4,3.0e-4,4.0e-4,5.5e-4,7.5e-4,1.1e-3,1.8e-3,8e-3",
    },
    {
        "tag": "A1_feat_strong",
        "layout": "15",
        "emb": "160",
        "dfeat": "16",
        "dexp": "256",
        "drouter": "112",
        "scale": "4",
        "train_bs": "2048",
        "eval_bs": "4096",
        "outer_feature": "true",
        "inner_mode": "distill",
        "inner_lambda": "5e-3",
        "inner_tau": "1.2",
        "inner_until": "0.4",
        "inner_bias": "0.0",
        "expert_top_k": "1",
        "base_lr": "3.5e-4",
        "lr_space": "1e-4,1.5e-4,2.0e-4,2.8e-4,3.5e-4,5.0e-4,7.0e-4,1.0e-3,1.8e-3,8e-3",
    },
    {
        "tag": "A1_honly_strong",
        "layout": "15",
        "emb": "160",
        "dfeat": "16",
        "dexp": "256",
        "drouter": "112",
        "scale": "4",
        "train_bs": "2048",
        "eval_bs": "4096",
        "outer_feature": "false",
        "inner_mode": "distill",
        "inner_lambda": "5e-3",
        "inner_tau": "1.2",
        "inner_until": "0.4",
        "inner_bias": "0.0",
        "expert_top_k": "1",
        "base_lr": "3.5e-4",
        "lr_space": "1e-4,1.5e-4,2.0e-4,2.8e-4,3.5e-4,5.0e-4,7.0e-4,1.0e-3,1.8e-3,8e-3",
    },
]

for combo in combos:
    row = [
        combo["tag"],
        combo["layout"],
        combo["emb"],
        combo["dfeat"],
        combo["dexp"],
        combo["drouter"],
        combo["scale"],
        combo["train_bs"],
        combo["eval_bs"],
        combo["outer_feature"],
        combo["inner_mode"],
        combo["inner_lambda"],
        combo["inner_tau"],
        combo["inner_until"],
        combo["inner_bias"],
        combo["expert_top_k"],
        combo["base_lr"],
        combo["lr_space"],
    ]
    print("\t".join(row))
PY
}

read_combo() {
  local idx="$1"
  generate_combo_rows | sed -n "$((idx + 1))p"
}

combo_count() {
  generate_combo_rows | wc -l | tr -d ' '
}

combo_total="$(combo_count)"
echo "[R2O-HGRv3] combo_count=${combo_total} combos_per_gpu=${COMBOS_PER_GPU} gpus=${GPU_LIST}"

for ds in "${DATASET_ARR[@]}"; do
  for gidx in "${!GPUS[@]}"; do
    gpu="${GPUS[$gidx]}"
    (
      set -euo pipefail
      for slot in $(seq 0 $((COMBOS_PER_GPU - 1))); do
        idx=$(( gidx * COMBOS_PER_GPU + slot ))
        if [ "$idx" -ge "$combo_total" ]; then
          continue
        fi
        combo_id="$idx"
        seed=$(( SEED_BASE + idx ))
        combo_row="$(read_combo "$combo_id")"
        IFS=$'\t' read -r combo_tag layout_id emb dfeat dexp drouter scale train_bs eval_bs outer_feature inner_mode inner_lambda inner_tau inner_until inner_bias expert_top_k base_lr lr_space <<< "$combo_row"

        phase="${PHASE_PREFIX}_C$(printf '%02d' "$combo_id")_${combo_tag}"
        exp_desc="${EXP_DESC_BASE} layout=${layout_id} outer_feature=${outer_feature} inner_mode=${inner_mode} lambda=${inner_lambda} tau=${inner_tau} until=${inner_until} dims=${emb}/${dfeat}/${dexp}/${drouter} bs=${train_bs}/${eval_bs}"

        cmd=(
          bash "${SCRIPT_DIR}/tune_hparam.sh"
          --dataset "${ds}"
          --gpu "${gpu}"
          --max-evals "${MAX_EVALS}"
          --tune-epochs "${TUNE_EPOCHS}"
          --tune-patience "${TUNE_PATIENCE}"
          --seed "${seed}"
          --phase "${phase}"
          --layout-id "${layout_id}"
          --train-batch-size "${train_bs}"
          --eval-batch-size "${eval_bs}"
          --embedding-size "${emb}"
          --d-feat-emb "${dfeat}"
          --d-expert-hidden "${dexp}"
          --d-router-hidden "${drouter}"
          --expert-scale "${scale}"
          --group-top-k "0"
          --expert-top-k "${expert_top_k}"
          --outer-router-use-hidden "true"
          --outer-router-use-feature "${outer_feature}"
          --learning-rate "${base_lr}"
          --weight-decay "1e-5"
          --dropout "0.10"
          --balance-loss-lambda "0.0"
          --group-balance-lambda "0.0"
          --intra-balance-lambda "0.0"
          --group-feature-spec-aux-lambda "1e-4"
          --inner-rule-mode "${inner_mode}"
          --inner-rule-lambda "${inner_lambda}"
          --inner-rule-temperature "${inner_tau}"
          --inner-rule-until "${inner_until}"
          --inner-rule-bias-scale "${inner_bias}"
          --inner-rule-bin-sharpness "16.0"
          --inner-rule-apply-stages "[macro,mid,micro]"
          --lr-space "${lr_space}"
          --wd-space "1e-5"
          --dropout-space "0.10"
          --balance-space "0.0"
          --exp-name "${EXP_NAME_BASE}_${combo_tag}"
          --exp-desc "${exp_desc}"
          --exp-focus "${EXP_FOCUS}"
        )
        if [ "$LOG_WANDB" = "true" ]; then
          cmd+=(--log-wandb)
        fi
        if [ "$DRY_RUN" = "true" ]; then
          cmd+=(--dry-run)
        fi
        run_echo_cmd "${cmd[@]}"
        "${cmd[@]}"
      done
    ) &
  done
  wait
done
