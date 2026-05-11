#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=/dev/null
source "${RUN_DIR}/common/run_metadata.sh"
# shellcheck source=/dev/null
source "${RUN_DIR}/common/dispatch_gpu_queue.sh"

DATASET=""
PARENT_RESULT=""
GPU_LIST="0,1"
COMBOS_PER_GPU="2"
MAX_EVALS="20"
TUNE_EPOCHS="100"
TUNE_PATIENCE="10"
SEED_BASE="1080"
PHASE_PREFIX="P3HGR"
LOG_WANDB="false"
DRY_RUN="${DRY_RUN:-false}"
WD_SPACE="0,1e-6,1e-5,1e-4"
DROP_SPACE="0.08,0.12,0.16"
BAL_SPACE="0.003,0.01,0.03"
EXP_NAME_BASE="P3_hgr_dim_batch"
EXP_DESC_BASE="HGR structure refinement around best routing combo; each combo re-tunes optimizer and regularization."
EXP_FOCUS="stage_merge_mode,group_router_mode,arch_layout_id,group_top_k,expert_use_feature,macro_routing_scope,parallel_stage_gate_temperature,embedding_size,d_feat_emb,d_expert_hidden,d_router_hidden,expert_scale,train_batch_size,eval_batch_size,learning_rate,weight_decay,hidden_dropout_prob,balance_loss_lambda"

# emb,d_feat,d_exp,d_router,expert_scale,train_bs,eval_bs
COMBO_CATALOG="\
128,16,160,64,3,4096,8192;\
128,16,192,64,3,4096,8192;\
128,24,192,80,3,3072,6144;\
160,16,192,80,3,3072,6144;\
160,24,224,96,3,3072,6144;\
128,16,160,64,4,2048,4096"

usage() {
  cat <<USAGE
Usage: $0 --dataset <ds> --parent-result <json> [--gpus 0,1]
          [--combos-per-gpu N] [--max-evals N] [--combo-catalog spec]
USAGE
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2 ;;
    --parent-result|--parent_result) PARENT_RESULT="$2"; shift 2 ;;
    --gpus) GPU_LIST="$2"; shift 2 ;;
    --combos-per-gpu) COMBOS_PER_GPU="$2"; shift 2 ;;
    --max-evals) MAX_EVALS="$2"; shift 2 ;;
    --tune-epochs) TUNE_EPOCHS="$2"; shift 2 ;;
    --tune-patience) TUNE_PATIENCE="$2"; shift 2 ;;
    --seed-base) SEED_BASE="$2"; shift 2 ;;
    --phase-prefix) PHASE_PREFIX="$2"; shift 2 ;;
    --combo-catalog) COMBO_CATALOG="$2"; shift 2 ;;
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

[ -z "$DATASET" ] && { echo "--dataset required" >&2; exit 1; }
[ -z "$PARENT_RESULT" ] && { echo "--parent-result required" >&2; exit 1; }
[ ! -f "$PARENT_RESULT" ] && { echo "parent result not found: $PARENT_RESULT" >&2; exit 1; }

dispatch_parse_csv "$GPU_LIST" GPUS
[ "${#GPUS[@]}" -eq 0 ] && { echo "Empty GPU list" >&2; exit 1; }

read_combo() {
  local idx="$1"
  python3 - <<'PY' "$COMBO_CATALOG" "$idx"
import sys
rows = [x.strip() for x in sys.argv[1].split(';') if x.strip()]
if not rows:
    raise SystemExit("empty combo catalog")
row = rows[int(sys.argv[2]) % len(rows)]
parts = [x.strip() for x in row.split(',')]
if len(parts) != 7:
    raise SystemExit(f"invalid combo row: {row}")
print(" ".join(parts))
PY
}

combo_count() {
  python3 - <<'PY' "$COMBO_CATALOG"
import sys
rows = [x.strip() for x in sys.argv[1].split(';') if x.strip()]
print(len(rows))
PY
}

lr_space_for_bs() {
  local train_bs="$1"
  if [ "$train_bs" -le 2048 ]; then
    echo "2e-4,4e-4,7.5e-4,1e-3,2e-3,3e-3,5e-3"
  elif [ "$train_bs" -le 3072 ]; then
    echo "2.5e-4,5e-4,1e-3,2e-3,3e-3,5e-3,7.5e-3"
  else
    echo "3e-4,7.5e-4,1e-3,2e-3,3e-3,5e-3,8e-3"
  fi
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
  if { [ "$emb" -ge 160 ] || [ "$d_exp" -ge 224 ] || [ "$d_router" -ge 96 ]; } && [ "$capped_train" -gt 3072 ]; then
    capped_train=3072
    capped_eval=6144
  fi
  if [ "$expert_scale" -ge 4 ] && [ "$capped_train" -gt 2048 ]; then
    capped_train=2048
    capped_eval=4096
  fi
  echo "${capped_train} ${capped_eval}"
}

INTERRUPTED="false"
WORKER_PIDS=()
on_interrupt() {
  INTERRUPTED="true"
  echo "[INTERRUPT] stopping HGR P3 workers..."
  local p
  for p in "${WORKER_PIDS[@]:-}"; do
    kill -TERM "$p" 2>/dev/null || true
  done
  wait || true
  exit 130
}
trap on_interrupt INT TERM

combo_total="$(combo_count)"
echo "[P3-HGR] combo_count=${combo_total} combos_per_gpu=${COMBOS_PER_GPU} gpus=${GPU_LIST}"

WORKER_PIDS=()
for gidx in "${!GPUS[@]}"; do
  gpu="${GPUS[$gidx]}"
  (
    set -euo pipefail
    for slot in $(seq 0 $((COMBOS_PER_GPU - 1))); do
      idx=$(( gidx * COMBOS_PER_GPU + slot ))
      seed=$(( SEED_BASE + idx ))
      combo_id=$(( idx % combo_total ))
      read -r emb dfeat dexp drouter scale train_bs eval_bs <<< "$(read_combo "$combo_id")"
      read -r safe_train_bs safe_eval_bs <<< "$(apply_oom_safety_cap "$emb" "$dexp" "$drouter" "$scale" "$train_bs" "$eval_bs")"
      lr_space="$(lr_space_for_bs "$safe_train_bs")"
      phase="${PHASE_PREFIX}_C$(printf '%02d' "$combo_id")_E${emb}_R${drouter}"
      exp_name="${EXP_NAME_BASE}"
      exp_desc="${EXP_DESC_BASE} combo=C${combo_id} emb=${emb} dfeat=${dfeat} dexp=${dexp} drouter=${drouter} scale=${scale} bs=${safe_train_bs}/${safe_eval_bs}"

      cmd=(
        bash "${SCRIPT_DIR}/tune_hparam.sh"
        --dataset "${DATASET}"
        --gpu "${gpu}"
        --max-evals "${MAX_EVALS}"
        --tune-epochs "${TUNE_EPOCHS}"
        --tune-patience "${TUNE_PATIENCE}"
        --seed "${seed}"
        --phase "${phase}"
        --search-profile structure_refine
        --parent-result "${PARENT_RESULT}"
        --embedding-size "${emb}"
        --d-feat-emb "${dfeat}"
        --d-expert-hidden "${dexp}"
        --d-router-hidden "${drouter}"
        --expert-scale "${scale}"
        --train-batch-size "${safe_train_bs}"
        --eval-batch-size "${safe_eval_bs}"
        --lr-space "${lr_space}"
        --wd-space "${WD_SPACE}"
        --dropout-space "${DROP_SPACE}"
        --balance-space "${BAL_SPACE}"
        --exp-name "${exp_name}"
        --exp-desc "${exp_desc}"
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

      echo "[P3-HGR] gpu=${gpu} slot=${slot} combo=C${combo_id} emb=${emb} dfeat=${dfeat} dexp=${dexp} drouter=${drouter} scale=${scale} bs=${safe_train_bs}/${safe_eval_bs}"
      "${cmd[@]}"
    done
  ) &
  WORKER_PIDS+=("$!")
done

wait
if [ "$INTERRUPTED" = "true" ]; then
  exit 130
fi

run_update_track_report fmoe_hgr
