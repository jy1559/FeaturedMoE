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
COMBOS_PER_GPU="6"
MAX_EVALS="8"
TUNE_EPOCHS="25"
TUNE_PATIENCE="5"
SEED_BASE="520"
PHASE_PREFIX="P15HGR_layout24"
LOG_WANDB="false"
DRY_RUN="${DRY_RUN:-false}"
LR_SPACE="5e-4,8e-4,1.1e-3,1.5e-3,2.0e-3,2.6e-3,3.2e-3"
WD_SPACE="0,1e-6,3e-6,1e-5,3e-5"
DROP_SPACE="0.08,0.10,0.12"
BAL_SPACE="0.0015,0.003,0.006"
GROUP_BAL="0.001"
INTRA_BAL="0.001"
SPEC_LAMBDA="1e-4"
EXP_NAME_BASE="P15_hgr_layout_focus"
EXP_DESC_BASE="Layout-focused HGR P1.5 screen. Uses widewide top layouts as anchors, keeps aux routing regularizers fixed, and only allows small dim variation so structure ranking is cleaner before P2."
EXP_FOCUS="arch_layout_id,num_layers,stage_merge_mode,group_router_mode,embedding_size,d_feat_emb,d_expert_hidden,d_router_hidden,expert_scale,learning_rate,weight_decay,hidden_dropout_prob,balance_loss_lambda"

usage() {
  cat <<USAGE
Usage: $0 [--datasets movielens1m] [--gpus 0,1,2,3]
          [--combos-per-gpu 6] [--max-evals 8] [--tune-epochs 25] [--tune-patience 5]
          [--lr-space csv] [--wd-space csv] [--dropout-space csv] [--balance-space csv]
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
    --lr-space) LR_SPACE="$2"; shift 2 ;;
    --wd-space) WD_SPACE="$2"; shift 2 ;;
    --dropout-space) DROP_SPACE="$2"; shift 2 ;;
    --balance-space) BAL_SPACE="$2"; shift 2 ;;
    --group-balance-lambda) GROUP_BAL="$2"; shift 2 ;;
    --intra-balance-lambda) INTRA_BAL="$2"; shift 2 ;;
    --group-feature-spec-aux-lambda) SPEC_LAMBDA="$2"; shift 2 ;;
    --exp-name-base) EXP_NAME_BASE="$2"; shift 2 ;;
    --exp-desc-base) EXP_DESC_BASE="$2"; shift 2 ;;
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
if ! [[ "$MAX_EVALS" =~ ^[0-9]+$ ]] || [ "$MAX_EVALS" -le 0 ]; then
  echo "--max-evals must be positive integer" >&2
  exit 1
fi

dispatch_parse_csv "$GPU_LIST" GPUS
[ "${#GPUS[@]}" -eq 0 ] && { echo "Empty GPU list" >&2; exit 1; }
dispatch_parse_csv "$DATASETS" DATASET_ARR
[ "${#DATASET_ARR[@]}" -eq 0 ] && { echo "Empty dataset list" >&2; exit 1; }

generate_combo_rows() {
  python3 - <<'PY'
arch_profiles = [
    ("L0A", 0, 4, 128, 16, 160, 64, 3, 4096, 8192),
    ("L0W", 0, 4, 160, 16, 224, 96, 3, 3072, 6144),
    ("L5A", 5, 4, 128, 16, 160, 64, 3, 4096, 8192),
    ("L11A", 11, 3, 128, 16, 160, 64, 3, 4096, 8192),
    ("L15A", 15, 4, 128, 16, 160, 64, 3, 4096, 8192),
    ("L16A", 16, 5, 128, 16, 160, 64, 3, 4096, 8192),
    ("L10A", 10, 3, 128, 16, 160, 64, 3, 4096, 8192),
    ("L21W", 21, 7, 160, 16, 224, 96, 3, 2048, 4096),
]

route_profiles = [
    ("R0", "serial", "per_group", 0, 0, "off"),
    ("R1", "serial", "per_group", 0, 0, "alpha_cold"),
    ("R2", "serial", "hybrid", 0, 0, "off"),
]

for arch_tag, layout_id, total_layers, emb, d_feat, d_exp, d_router, expert_scale, train_bs, eval_bs in arch_profiles:
    for route_tag, merge_mode, group_mode, group_top_k, moe_top_k, sched in route_profiles:
        print(
            ",".join(
                str(x)
                for x in (
                    arch_tag,
                    route_tag,
                    layout_id,
                    total_layers,
                    merge_mode,
                    group_mode,
                    group_top_k,
                    moe_top_k,
                    emb,
                    d_feat,
                    d_exp,
                    d_router,
                    expert_scale,
                    train_bs,
                    eval_bs,
                    sched,
                )
            )
        )
PY
}

read_combo() {
  local idx="$1"
  generate_combo_rows | sed -n "$((idx + 1))p"
}

combo_count() {
  generate_combo_rows | wc -l | tr -d ' '
}

INTERRUPTED="false"
WORKER_PIDS=()
on_interrupt() {
  INTERRUPTED="true"
  echo "[INTERRUPT] stopping HGR P1.5 workers..."
  local p
  for p in "${WORKER_PIDS[@]:-}"; do
    kill -TERM "$p" 2>/dev/null || true
  done
  wait || true
  exit 130
}
trap on_interrupt INT TERM

combo_total="$(combo_count)"
echo "[P15-HGR] combo_count=${combo_total} layout_profiles=8 route_profiles=3 combos_per_gpu=${COMBOS_PER_GPU} gpus=${GPU_LIST}"

for ds in "${DATASET_ARR[@]}"; do
  total_jobs=$(( ${#GPUS[@]} * COMBOS_PER_GPU ))
  echo "=== [${ds}] HGR P1.5 layout-focused screen (${total_jobs} runs = ${#GPUS[@]} gpus x ${COMBOS_PER_GPU}) ==="
  if [ "$total_jobs" -lt "$combo_total" ]; then
    echo "[P15-HGR] warning: total_jobs=${total_jobs} < combo_total=${combo_total}; only the first combos will run."
  elif [ "$total_jobs" -gt "$combo_total" ]; then
    echo "[P15-HGR] warning: total_jobs=${total_jobs} > combo_total=${combo_total}; some combos will repeat."
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
        [ -n "$combo_row" ] || { echo "[P15-HGR] missing combo row for idx=${combo_id}" >&2; exit 1; }

        IFS=, read -r arch_tag route_tag layout_id total_layers merge_mode group_mode group_topk moe_topk emb dfeat dexp drouter scale train_bs eval_bs schedule_preset <<< "$combo_row"

        phase="${PHASE_PREFIX}_C$(printf '%02d' "$combo_id")_${arch_tag}_${route_tag}_${merge_mode}_${group_mode}"
        exp_name="${EXP_NAME_BASE}"
        exp_desc="${EXP_DESC_BASE} combo=C${combo_id} arch=${arch_tag} layout=${layout_id} total_layers=${total_layers} merge=${merge_mode} group=${group_mode} sched=${schedule_preset}"

        cmd=(
          bash "${SCRIPT_DIR}/tune_hparam.sh"
          --dataset "${ds}"
          --gpu "${gpu}"
          --max-evals "${MAX_EVALS}"
          --tune-epochs "${TUNE_EPOCHS}"
          --tune-patience "${TUNE_PATIENCE}"
          --seed "${seed}"
          --phase "${phase}"
          --search-profile wide
          --schedule-preset "${schedule_preset}"
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
          --train-batch-size "${train_bs}"
          --eval-batch-size "${eval_bs}"
          --embedding-size "${emb}"
          --d-feat-emb "${dfeat}"
          --d-expert-hidden "${dexp}"
          --d-router-hidden "${drouter}"
          --expert-scale "${scale}"
          --group-balance-lambda "${GROUP_BAL}"
          --intra-balance-lambda "${INTRA_BAL}"
          --group-feature-spec-aux-enable "true"
          --group-feature-spec-aux-lambda "${SPEC_LAMBDA}"
          --group-feature-spec-stages "[mid]"
          --group-feature-spec-min-tokens "8"
          --router-distill-enable "false"
          --lr-space "${LR_SPACE}"
          --wd-space "${WD_SPACE}"
          --dropout-space "${DROP_SPACE}"
          --balance-space "${BAL_SPACE}"
          --override "++search.group_balance_lambda=[${GROUP_BAL}]"
          --override "++search.intra_balance_lambda=[${INTRA_BAL}]"
          --override "++search.group_feature_spec_aux_lambda=[${SPEC_LAMBDA}]"
          --override "++search.group_feature_spec_aux_enable=[true]"
          --override "++search.group_feature_spec_stages=[[mid]]"
          --override "++search.group_feature_spec_min_tokens=[8]"
          --override "++search.router_distill_enable=[false]"
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

        echo "[P15-HGR] gpu=${gpu} slot=${slot} combo=C${combo_id} arch=${arch_tag} layout=${layout_id}/${total_layers}L merge=${merge_mode} group=${group_mode} bs=${train_bs}/${eval_bs} emb=${emb} feat=${dfeat} exp=${dexp} router=${drouter} scale=${scale} sched=${schedule_preset}"
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
