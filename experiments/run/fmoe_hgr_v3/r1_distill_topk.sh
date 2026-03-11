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
COMBOS_PER_GPU="2"
MAX_EVALS="8"
TUNE_EPOCHS="50"
TUNE_PATIENCE="10"
SEED_BASE="2410"
PHASE_PREFIX="R1DHGRv3_topk8"
LOG_WANDB="false"
DRY_RUN="${DRY_RUN:-false}"
EXP_NAME_BASE="R1_hgr_v3_distill_topk"
EXP_DESC_BASE="HGRv3 R1 distill-focused sweep: fixed distill family with expert_top_k and distill strength variations."
EXP_FOCUS="arch_layout_id,embedding_size,d_expert_hidden,d_router_hidden,expert_scale,expert_top_k,inner_rule_mode,inner_rule_lambda,inner_rule_until,learning_rate,weight_decay"

usage() {
  cat <<USAGE
Usage: $0 [--datasets movielens1m] [--gpus 0,1,2,3]
          [--combos-per-gpu 2] [--max-evals 8] [--tune-epochs 50] [--tune-patience 10]
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
anchors = [
    {
        "anchor": "A0",
        "emb": 128,
        "dfeat": 16,
        "dexp": 160,
        "drouter": 64,
        "scale": 4,
        "train_bs": 4096,
        "eval_bs": 8192,
        "base_wd": "1e-6",
    },
    {
        "anchor": "A1",
        "emb": 160,
        "dfeat": 16,
        "dexp": 256,
        "drouter": 112,
        "scale": 4,
        "train_bs": 2048,
        "eval_bs": 4096,
        "base_wd": "1e-6",
    },
]

combos = [
    {"anchor": "A0", "tag": "weak_k1", "mode": "distill", "lambda": "2e-3", "tau": "1.5", "until": "0.2", "bias": "0.0", "expert_top_k": "1", "base_lr": "1.7e-3", "lr_space": "1e-4,4e-4,7e-4,1.0e-3,1.3e-3,1.7e-3,2.2e-3,3.0e-3,4.5e-3,8e-3"},
    {"anchor": "A0", "tag": "weak_k2", "mode": "distill", "lambda": "2e-3", "tau": "1.5", "until": "0.2", "bias": "0.0", "expert_top_k": "2", "base_lr": "1.7e-3", "lr_space": "1e-4,4e-4,7e-4,1.0e-3,1.3e-3,1.7e-3,2.2e-3,3.0e-3,4.5e-3,8e-3"},
    {"anchor": "A0", "tag": "weak_k4", "mode": "distill", "lambda": "2e-3", "tau": "1.5", "until": "0.2", "bias": "0.0", "expert_top_k": "4", "base_lr": "1.5e-3", "lr_space": "1e-4,3e-4,5e-4,8e-4,1.1e-3,1.5e-3,2.0e-3,2.8e-3,4.0e-3,8e-3"},
    {"anchor": "A0", "tag": "main_k2", "mode": "distill", "lambda": "5e-3", "tau": "1.5", "until": "0.2", "bias": "0.0", "expert_top_k": "2", "base_lr": "1.6e-3", "lr_space": "1e-4,3.5e-4,6e-4,9e-4,1.2e-3,1.6e-3,2.0e-3,2.7e-3,4.0e-3,8e-3"},
    {"anchor": "A1", "tag": "weak_k1", "mode": "distill", "lambda": "2e-3", "tau": "1.5", "until": "0.2", "bias": "0.0", "expert_top_k": "1", "base_lr": "5e-4", "lr_space": "1e-4,1.8e-4,2.5e-4,3.5e-4,5.0e-4,7.0e-4,1.0e-3,1.5e-3,2.5e-3,8e-3"},
    {"anchor": "A1", "tag": "weak_k2", "mode": "distill", "lambda": "2e-3", "tau": "1.5", "until": "0.2", "bias": "0.0", "expert_top_k": "2", "base_lr": "5e-4", "lr_space": "1e-4,1.8e-4,2.5e-4,3.5e-4,5.0e-4,7.0e-4,1.0e-3,1.5e-3,2.5e-3,8e-3"},
    {"anchor": "A1", "tag": "weak_k4", "mode": "distill", "lambda": "2e-3", "tau": "1.5", "until": "0.2", "bias": "0.0", "expert_top_k": "4", "base_lr": "3.8e-4", "lr_space": "1e-4,1.5e-4,2.0e-4,2.8e-4,3.8e-4,5.0e-4,7.0e-4,1.0e-3,1.8e-3,8e-3"},
    {"anchor": "A1", "tag": "main_k2", "mode": "distill", "lambda": "5e-3", "tau": "1.5", "until": "0.2", "bias": "0.0", "expert_top_k": "2", "base_lr": "4e-4", "lr_space": "1e-4,1.5e-4,2.2e-4,3.0e-4,4.0e-4,5.5e-4,7.5e-4,1.1e-3,1.8e-3,8e-3"},
]

anchor_map = {anchor["anchor"]: anchor for anchor in anchors}
for combo in combos:
    anchor = anchor_map[combo["anchor"]]
    row = [
        f"{anchor['anchor']}_{combo['tag']}",
        anchor["anchor"],
        combo["tag"],
        anchor["emb"],
        anchor["dfeat"],
        anchor["dexp"],
        anchor["drouter"],
        anchor["scale"],
        anchor["train_bs"],
        anchor["eval_bs"],
        combo["base_lr"],
        anchor["base_wd"],
        combo["lr_space"],
        "1e-6",
        combo["mode"],
        combo["lambda"],
        combo["tau"],
        combo["until"],
        combo["bias"],
        combo["expert_top_k"],
    ]
    print("\t".join(str(x) for x in row))
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
echo "[R1D-HGRv3] combo_count=${combo_total} combos_per_gpu=${COMBOS_PER_GPU} gpus=${GPU_LIST}"

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
        IFS=$'\t' read -r combo_tag anchor_tag mode_tag emb dfeat dexp drouter scale train_bs eval_bs base_lr base_wd lr_space wd_space inner_mode inner_lambda inner_tau inner_until inner_bias expert_top_k <<< "$combo_row"

        phase="${PHASE_PREFIX}_C$(printf '%02d' "$combo_id")_${combo_tag}"
        exp_desc="${EXP_DESC_BASE} anchor=${anchor_tag} mode=${mode_tag} k=${expert_top_k} lambda=${inner_lambda} dims=${emb}/${dfeat}/${dexp}/${drouter} scale=${scale} bs=${train_bs}/${eval_bs}"

        cmd=(
          bash "${SCRIPT_DIR}/tune_hparam.sh"
          --dataset "${ds}"
          --gpu "${gpu}"
          --max-evals "${MAX_EVALS}"
          --tune-epochs "${TUNE_EPOCHS}"
          --tune-patience "${TUNE_PATIENCE}"
          --seed "${seed}"
          --phase "${phase}"
          --layout-id "15"
          --train-batch-size "${train_bs}"
          --eval-batch-size "${eval_bs}"
          --embedding-size "${emb}"
          --d-feat-emb "${dfeat}"
          --d-expert-hidden "${dexp}"
          --d-router-hidden "${drouter}"
          --expert-scale "${scale}"
          --group-top-k "0"
          --expert-top-k "${expert_top_k}"
          --learning-rate "${base_lr}"
          --weight-decay "${base_wd}"
          --dropout "0.10"
          --balance-loss-lambda "0.003"
          --group-balance-lambda "0.001"
          --intra-balance-lambda "0.001"
          --group-feature-spec-aux-lambda "1e-4"
          --inner-rule-mode "${inner_mode}"
          --inner-rule-lambda "${inner_lambda}"
          --inner-rule-temperature "${inner_tau}"
          --inner-rule-until "${inner_until}"
          --inner-rule-bias-scale "${inner_bias}"
          --inner-rule-bin-sharpness "16.0"
          --inner-rule-apply-stages "[macro,mid,micro]"
          --lr-space "${lr_space}"
          --wd-space "${wd_space}"
          --dropout-space "0.10"
          --balance-space "0.003"
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
