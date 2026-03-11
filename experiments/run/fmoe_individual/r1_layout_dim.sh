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
MAX_EVALS="8"
TUNE_EPOCHS="50"
TUNE_PATIENCE="10"
SEED_BASE="3110"
PHASE_PREFIX="R1INDv1_layout4"
LOG_WANDB="false"
DRY_RUN="${DRY_RUN:-false}"
EXP_NAME_BASE="R1_individual_layout_dim"
EXP_DESC_BASE="FeaturedMoE_Individual layout/dim hyperopt sweep with strict round-robin GPU assignment."
EXP_FOCUS="arch_layout_id,embedding_size,d_feat_emb,d_expert_hidden,d_router_hidden,learning_rate,weight_decay,hidden_dropout_prob"

usage() {
  cat <<USAGE
Usage: $0 [--datasets movielens1m] [--gpus 0,1,2,3]
          [--max-evals 8] [--tune-epochs 50] [--tune-patience 10]
USAGE
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --datasets) DATASETS="$2"; shift 2 ;;
    --gpus) GPU_LIST="$2"; shift 2 ;;
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
    {"tag":"D0_L0", "layout":"0", "emb":"128", "dfeat":"16", "dexp":"160", "drouter":"64",  "train_bs":"4096", "eval_bs":"8192"},
    {"tag":"D0_L1", "layout":"1", "emb":"128", "dfeat":"16", "dexp":"160", "drouter":"64",  "train_bs":"4096", "eval_bs":"8192"},
    {"tag":"D0_L2", "layout":"2", "emb":"128", "dfeat":"16", "dexp":"160", "drouter":"64",  "train_bs":"4096", "eval_bs":"8192"},
    {"tag":"D0_L3", "layout":"3", "emb":"128", "dfeat":"16", "dexp":"160", "drouter":"64",  "train_bs":"4096", "eval_bs":"8192"},
    {"tag":"D1_L0", "layout":"0", "emb":"128", "dfeat":"24", "dexp":"192", "drouter":"80",  "train_bs":"3072", "eval_bs":"6144"},
    {"tag":"D1_L1", "layout":"1", "emb":"128", "dfeat":"24", "dexp":"192", "drouter":"80",  "train_bs":"3072", "eval_bs":"6144"},
    {"tag":"D1_L2", "layout":"2", "emb":"128", "dfeat":"24", "dexp":"192", "drouter":"80",  "train_bs":"3072", "eval_bs":"6144"},
    {"tag":"D1_L3", "layout":"3", "emb":"128", "dfeat":"24", "dexp":"192", "drouter":"80",  "train_bs":"3072", "eval_bs":"6144"},
    {"tag":"D2_L0", "layout":"0", "emb":"160", "dfeat":"16", "dexp":"256", "drouter":"112", "train_bs":"2048", "eval_bs":"4096"},
    {"tag":"D2_L1", "layout":"1", "emb":"160", "dfeat":"16", "dexp":"256", "drouter":"112", "train_bs":"2048", "eval_bs":"4096"},
    {"tag":"D2_L2", "layout":"2", "emb":"160", "dfeat":"16", "dexp":"256", "drouter":"112", "train_bs":"2048", "eval_bs":"4096"},
    {"tag":"D2_L3", "layout":"3", "emb":"160", "dfeat":"16", "dexp":"256", "drouter":"112", "train_bs":"2048", "eval_bs":"4096"},
]

for combo in combos:
    row = [
        combo["tag"],
        combo["layout"],
        combo["emb"],
        combo["dfeat"],
        combo["dexp"],
        combo["drouter"],
        combo["train_bs"],
        combo["eval_bs"],
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
gpu_n="${#GPUS[@]}"
echo "[R1-Individual] combo_count=${combo_total} gpu_count=${gpu_n} gpus=${GPU_LIST}"

for ds in "${DATASET_ARR[@]}"; do
  for gidx in "${!GPUS[@]}"; do
    gpu="${GPUS[$gidx]}"
    (
      set -euo pipefail
      idx="$gidx"
      while [ "$idx" -lt "$combo_total" ]; do
        combo_row="$(read_combo "$idx")"
        IFS=$'\t' read -r combo_tag layout_id emb dfeat dexp drouter train_bs eval_bs <<< "$combo_row"
        seed=$(( SEED_BASE + idx ))
        phase="${PHASE_PREFIX}_C$(printf '%02d' "$idx")_${combo_tag}"
        exp_desc="${EXP_DESC_BASE} layout=${layout_id} dims=${emb}/${dfeat}/${dexp}/${drouter} bs=${train_bs}/${eval_bs}"

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
          --expert-scale "4"
          --feature-top-k "4"
          --inner-expert-top-k "0"
          --learning-rate "1.8e-3"
          --weight-decay "1e-6"
          --dropout "0.10"
          --lr-space "1e-4,2e-4,3.5e-4,6e-4,9e-4,1.3e-3,1.8e-3,2.6e-3,4.0e-3,8e-3"
          --wd-space "1e-6"
          --dropout-space "0.1"
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
        idx=$(( idx + gpu_n ))
      done
    ) &
  done
  wait
done
