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
MAX_EVALS="6"
TUNE_EPOCHS="40"
TUNE_PATIENCE="8"
SEED_BASE="920"
PHASE_PREFIX="P2HGR_dim8"
LOG_WANDB="false"
DRY_RUN="${DRY_RUN:-false}"
DROP_SPACE="0.08,0.10,0.12"
BAL_SPACE="0.002,0.003,0.0045"
GROUP_BAL="0.001"
INTRA_BAL="0.001"
SPEC_LAMBDA="1e-4"
EXP_NAME_BASE="P2_hgr_dim_focus"
EXP_DESC_BASE="Post-P15 HGR P2 dim focus. Layout/route are nearly fixed from P15, and only dim/batch/LR coupling is probed."
EXP_FOCUS="arch_layout_id,stage_merge_mode,group_router_mode,embedding_size,d_feat_emb,d_expert_hidden,d_router_hidden,expert_scale,train_batch_size,eval_batch_size,learning_rate,weight_decay,hidden_dropout_prob,balance_loss_lambda"

usage() {
  cat <<USAGE
Usage: $0 [--datasets movielens1m] [--gpus 0,1,2,3]
          [--combos-per-gpu 2] [--max-evals 6] [--tune-epochs 40] [--tune-patience 8]
          [--dropout-space csv] [--balance-space csv]
          [--phase-prefix NAME] [--log-wandb] [--dry-run]
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
    --dropout-space) DROP_SPACE="$2"; shift 2 ;;
    --balance-space) BAL_SPACE="$2"; shift 2 ;;
    --group-balance-lambda) GROUP_BAL="$2"; shift 2 ;;
    --intra-balance-lambda) INTRA_BAL="$2"; shift 2 ;;
    --group-feature-spec-aux-lambda) SPEC_LAMBDA="$2"; shift 2 ;;
    --log-wandb) LOG_WANDB="true"; shift ;;
    --no-wandb) LOG_WANDB="false"; shift ;;
    --dry-run) DRY_RUN="true"; shift ;;
    --help|-h) usage; exit 0 ;;
    --) shift; break ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

if ! [[ "$COMBOS_PER_GPU" =~ ^[0-9]+$ ]] || [ "$COMBOS_PER_GPU" -le 0 ]; then
  echo "--combos-per-gpu must be positive integer" >&2
  exit 1
fi

dispatch_parse_csv "$GPU_LIST" GPUS
[ "${#GPUS[@]}" -eq 0 ] && { echo "Empty GPU list" >&2; exit 1; }
dispatch_parse_csv "$DATASETS" DATASET_ARR
[ "${#DATASET_ARR[@]}" -eq 0 ] && { echo "Empty dataset list" >&2; exit 1; }

generate_combo_rows() {
  python3 - <<'PY'
rows = [
    ("L15H_D0", 15, "serial", "hybrid", 0, 0, 128, 16, 160, 64, 3, 4096, 8192),
    ("L15H_D1", 15, "serial", "hybrid", 0, 0, 160, 16, 224, 96, 3, 3072, 6144),
    ("L15H_D2", 15, "serial", "hybrid", 0, 0, 160, 16, 256, 112, 3, 2048, 4096),
    ("L15H_D3", 15, "serial", "hybrid", 0, 0, 192, 16, 320, 128, 2, 1536, 3072),
    ("L15H_D4", 15, "serial", "hybrid", 0, 0, 224, 16, 512, 160, 1, 1024, 2048),
    ("L15P_D1", 15, "serial", "per_group", 0, 0, 160, 16, 224, 96, 3, 3072, 6144),
    ("L16H_D1", 16, "serial", "hybrid", 0, 0, 160, 16, 224, 96, 3, 3072, 6144),
    ("L21P_D3", 21, "serial", "per_group", 0, 0, 192, 16, 320, 128, 2, 1536, 3072),
]
for row in rows:
    print(",".join(str(x) for x in row))
PY
}

read_combo() {
  local idx="$1"
  generate_combo_rows | sed -n "$((idx + 1))p"
}

combo_count() {
  generate_combo_rows | wc -l | tr -d ' '
}

base_lr_for_bs() {
  local train_bs="$1"
  case "$train_bs" in
    4096) echo "1.5e-3" ;;
    3072) echo "1.1e-3" ;;
    2048) echo "8e-4" ;;
    1536) echo "8e-4" ;;
    1024) echo "7e-4" ;;
    *) echo "1e-3" ;;
  esac
}

base_wd_for_bs() {
  local train_bs="$1"
  case "$train_bs" in
    4096|3072) echo "1e-5" ;;
    2048|1536) echo "2e-5" ;;
    1024) echo "3e-5" ;;
    *) echo "1e-5" ;;
  esac
}

lr_space_for_bs() {
  local train_bs="$1"
  case "$train_bs" in
    4096) echo "8e-4,1.1e-3,1.5e-3,2.0e-3,2.6e-3,3.2e-3" ;;
    3072) echo "6e-4,8e-4,1.1e-3,1.5e-3,2.0e-3,2.6e-3" ;;
    2048) echo "4e-4,6e-4,8e-4,1.1e-3,1.5e-3,2.0e-3" ;;
    1536) echo "3e-4,4.5e-4,6e-4,8e-4,1.1e-3,1.5e-3" ;;
    1024) echo "2.5e-4,3.5e-4,5e-4,7e-4,9e-4,1.2e-3" ;;
    *) echo "6e-4,8e-4,1.1e-3,1.5e-3,2.0e-3" ;;
  esac
}

wd_space_for_bs() {
  local train_bs="$1"
  case "$train_bs" in
    4096|3072) echo "1e-6,3e-6,1e-5,2e-5,3e-5" ;;
    2048|1536) echo "3e-6,1e-5,2e-5,3e-5,5e-5" ;;
    1024) echo "1e-5,2e-5,3e-5,5e-5,8e-5" ;;
    *) echo "1e-6,1e-5,2e-5,3e-5" ;;
  esac
}

apply_oom_safety_cap() {
  local emb="$1"
  local d_exp="$2"
  local d_router="$3"
  local expert_scale="$4"
  local train_bs="$5"
  local eval_bs="$6"

  local capped_train="$train_bs"
  local capped_eval="$eval_bs"

  if { [ "$emb" -ge 224 ] || [ "$d_exp" -ge 512 ] || [ "$d_router" -ge 160 ]; } && [ "$capped_train" -gt 1024 ]; then
    capped_train=1024
    capped_eval=2048
  elif { [ "$emb" -ge 192 ] || [ "$d_exp" -ge 320 ] || [ "$d_router" -ge 128 ]; } && [ "$capped_train" -gt 1536 ]; then
    capped_train=1536
    capped_eval=3072
  elif { [ "$emb" -ge 160 ] || [ "$d_exp" -ge 256 ] || [ "$d_router" -ge 112 ] || [ "$expert_scale" -ge 4 ]; } && [ "$capped_train" -gt 2048 ]; then
    capped_train=2048
    capped_eval=4096
  fi

  echo "${capped_train} ${capped_eval}"
}

INTERRUPTED="false"
WORKER_PIDS=()
on_interrupt() {
  INTERRUPTED="true"
  echo "[INTERRUPT] stopping HGR P2 workers..."
  local p
  for p in "${WORKER_PIDS[@]:-}"; do
    kill -TERM "$p" 2>/dev/null || true
  done
  wait || true
  exit 130
}
trap on_interrupt INT TERM

combo_total="$(combo_count)"
echo "[P2-HGR] combo_count=${combo_total} combos_per_gpu=${COMBOS_PER_GPU} gpus=${GPU_LIST}"

for ds in "${DATASET_ARR[@]}"; do
  total_jobs=$(( ${#GPUS[@]} * COMBOS_PER_GPU ))
  echo "=== [${ds}] HGR P2 dim focus (${total_jobs} runs = ${#GPUS[@]} gpus x ${COMBOS_PER_GPU}) ==="
  if [ "$total_jobs" -lt "$combo_total" ]; then
    echo "[P2-HGR] warning: total_jobs=${total_jobs} < combo_total=${combo_total}; only the first combos will run."
  elif [ "$total_jobs" -gt "$combo_total" ]; then
    echo "[P2-HGR] warning: total_jobs=${total_jobs} > combo_total=${combo_total}; some combos will repeat."
  fi

  WORKER_PIDS=()
  for gidx in "${!GPUS[@]}"; do
    gpu="${GPUS[$gidx]}"
    (
      set -euo pipefail
      for slot in $(seq 0 $((COMBOS_PER_GPU - 1))); do
        idx=$(( gidx * COMBOS_PER_GPU + slot ))
        seed=$(( SEED_BASE + idx ))
        combo_id=$(( idx % combo_total ))
        combo_row="$(read_combo "$combo_id")"
        [ -n "$combo_row" ] || { echo "[P2-HGR] missing combo row for idx=${combo_id}" >&2; exit 1; }

        IFS=, read -r combo_tag layout_id merge_mode group_mode group_topk moe_topk emb dfeat dexp drouter scale train_bs eval_bs <<< "$combo_row"

        read -r safe_train_bs safe_eval_bs <<< "$(apply_oom_safety_cap "$emb" "$dexp" "$drouter" "$scale" "$train_bs" "$eval_bs")"
        lr_space="$(lr_space_for_bs "$safe_train_bs")"
        wd_space="$(wd_space_for_bs "$safe_train_bs")"
        base_lr="$(base_lr_for_bs "$safe_train_bs")"
        base_wd="$(base_wd_for_bs "$safe_train_bs")"

        phase="${PHASE_PREFIX}_C$(printf '%02d' "$combo_id")_${combo_tag}_${merge_mode}_${group_mode}"
        exp_desc="${EXP_DESC_BASE} combo=${combo_tag} layout=${layout_id} merge=${merge_mode} group=${group_mode} dims=${emb}/${dfeat}/${dexp}/${drouter} scale=${scale} bs=${safe_train_bs}/${safe_eval_bs}"

        cmd=(
          bash "${SCRIPT_DIR}/tune_hparam.sh"
          --dataset "${ds}"
          --gpu "${gpu}"
          --max-evals "${MAX_EVALS}"
          --tune-epochs "${TUNE_EPOCHS}"
          --tune-patience "${TUNE_PATIENCE}"
          --seed "${seed}"
          --phase "${phase}"
          --search-profile structure_refine
          --schedule-preset off
          --layout-id "${layout_id}"
          --stage-merge-mode "${merge_mode}"
          --group-router-mode "${group_mode}"
          --group-top-k "${group_topk}"
          --moe-top-k "${moe_topk}"
          --expert-top-k "1"
          --router-design "group_factorized_interaction"
          --expert-use-feature "false"
          --macro-routing-scope "session"
          --macro-session-pooling "query"
          --parallel-stage-gate-temperature "1.0"
          --train-batch-size "${safe_train_bs}"
          --eval-batch-size "${safe_eval_bs}"
          --embedding-size "${emb}"
          --d-feat-emb "${dfeat}"
          --d-expert-hidden "${dexp}"
          --d-router-hidden "${drouter}"
          --expert-scale "${scale}"
          --learning-rate "${base_lr}"
          --weight-decay "${base_wd}"
          --dropout "0.10"
          --balance-loss-lambda "0.003"
          --group-balance-lambda "${GROUP_BAL}"
          --intra-balance-lambda "${INTRA_BAL}"
          --group-feature-spec-aux-enable "true"
          --group-feature-spec-aux-lambda "${SPEC_LAMBDA}"
          --group-feature-spec-stages "[mid]"
          --group-feature-spec-min-tokens "8"
          --router-distill-enable "false"
          --lr-space "${lr_space}"
          --wd-space "${wd_space}"
          --dropout-space "${DROP_SPACE}"
          --balance-space "${BAL_SPACE}"
          --exp-name "${EXP_NAME_BASE}"
          --exp-desc "${exp_desc}"
          --exp-focus "${EXP_FOCUS}"
          --override "++search.group_balance_lambda=[${GROUP_BAL}]"
          --override "++search.intra_balance_lambda=[${INTRA_BAL}]"
          --override "++search.group_feature_spec_aux_lambda=[${SPEC_LAMBDA}]"
          --override "++search.group_feature_spec_aux_enable=[true]"
          --override "++search.group_feature_spec_stages=[[mid]]"
          --override "++search.group_feature_spec_min_tokens=[8]"
          --override "++search.router_distill_enable=[false]"
        )

        if [ "$LOG_WANDB" = "true" ]; then
          cmd+=(--log-wandb)
        else
          cmd+=(--no-wandb)
        fi
        if [ "$DRY_RUN" = "true" ]; then
          cmd+=(--dry-run)
        fi

        echo "[P2-HGR] gpu=${gpu} slot=${slot} combo=${combo_tag} layout=${layout_id} merge=${merge_mode} group=${group_mode} dims=${emb}/${dexp}/${drouter} bs=${safe_train_bs}/${safe_eval_bs}"
        "${cmd[@]}"
      done
    ) &
    WORKER_PIDS+=("$!")
  done

  wait
  if [ "$INTERRUPTED" = "true" ]; then
    exit 130
  fi
done

run_update_track_report fmoe_hgr
