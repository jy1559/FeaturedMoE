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
COMBOS_PER_GPU="8"
MAX_EVALS="8"
TUNE_EPOCHS="25"
TUNE_PATIENCE="5"
SEED_BASE="420"
PHASE_PREFIX="P1HGR_joint32"
LOG_WANDB="false"
DRY_RUN="${DRY_RUN:-false}"
LR_SPACE="2e-4,5e-4,1e-3,2e-3,4e-3,8e-3,1.5e-2"
WD_SPACE="0,1e-6,3e-6,1e-5,3e-5,1e-4"
DROP_SPACE="0.08,0.12,0.16"
BAL_SPACE="0.001,0.003,0.01"
EXP_NAME_BASE="P1_hgr_joint_fast32"
EXP_DESC_BASE="Fast ML1M HGR vNext screen after router redesign. Jointly sweeps layout depth and model capacity with 8 layout-capacity anchors x 4 routing/schedule profiles so structure and dimension effects are visible immediately."
EXP_FOCUS="arch_layout_id,num_layers,stage_merge_mode,group_router_mode,group_top_k,expert_top_k,expert_use_feature,macro_routing_scope,parallel_stage_gate_temperature,embedding_size,d_feat_emb,d_expert_hidden,d_router_hidden,expert_scale,train_batch_size,eval_batch_size,learning_rate,weight_decay,hidden_dropout_prob,balance_loss_lambda"

usage() {
  cat <<USAGE
Usage: $0 [--datasets movielens1m] [--gpus 0,1,2,3]
          [--combos-per-gpu 8] [--max-evals 8] [--tune-epochs 25] [--tune-patience 5]
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
    ("A0", 0, 4, 128, 16, 160, 64, 3, 4096, 8192, "false"),
    ("A1", 0, 4, 128, 16, 192, 80, 3, 4096, 8192, "false"),
    ("A2", 5, 4, 128, 16, 192, 80, 3, 4096, 8192, "false"),
    ("A3", 11, 3, 128, 16, 160, 64, 3, 6144, 12288, "false"),
    ("A4", 15, 4, 160, 16, 224, 96, 3, 3072, 6144, "false"),
    ("A5", 8, 2, 128, 16, 160, 64, 3, 8192, 16384, "false"),
    ("A6", 20, 6, 128, 24, 224, 96, 3, 3072, 6144, "true"),
    ("A7", 21, 7, 128, 24, 256, 128, 4, 2048, 4096, "true"),
]

route_profiles = [
    ("R0", "serial", "per_group", 0, 0, "session", "query", "1.0", "off"),
    ("R1", "serial", "per_group", 0, 0, "session", "query", "1.0", "alpha_cold"),
    ("R2", "serial", "hybrid", 0, 0, "session", "query", "1.0", "off"),
    ("R3", "parallel", "per_group", 0, 0, "session", "query", "0.9", "alpha_cold"),
]

for arch_tag, layout_id, total_layers, emb, d_feat, d_exp, d_router, expert_scale, train_bs, eval_bs, expert_feat in arch_profiles:
    for route_tag, merge_mode, group_mode, group_top_k, moe_top_k, macro_scope, macro_pool, par_temp, sched in route_profiles:
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
                    expert_feat,
                    macro_scope,
                    macro_pool,
                    par_temp,
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

apply_oom_safety_cap() {
  local dataset="$1"
  local total_layers="$2"
  local emb="$3"
  local d_exp="$4"
  local d_router="$5"
  local expert_scale="$6"
  local train_bs="$7"
  local eval_bs="$8"

  local capped_train="$train_bs"
  local capped_eval="$eval_bs"

  if [ "$dataset" = "retail_rocket" ] && [ "$capped_train" -gt 3072 ]; then
    capped_train=3072
    capped_eval=6144
  fi
  if { [ "$d_exp" -ge 224 ] || [ "$d_router" -ge 96 ]; } && [ "$capped_train" -gt 3072 ]; then
    capped_train=3072
    capped_eval=6144
  fi
  if [ "$expert_scale" -ge 4 ] && [ "$capped_train" -gt 2048 ]; then
    capped_train=2048
    capped_eval=4096
  fi
  if [ "$emb" -ge 160 ] && [ "$capped_train" -gt 3072 ]; then
    capped_train=3072
    capped_eval=6144
  fi
  if [ "$total_layers" -ge 7 ] && [ "$capped_train" -gt 2048 ]; then
    capped_train=2048
    capped_eval=4096
  fi

  echo "${capped_train} ${capped_eval}"
}

INTERRUPTED="false"
WORKER_PIDS=()
on_interrupt() {
  INTERRUPTED="true"
  echo "[INTERRUPT] stopping HGR P1 workers..."
  local p
  for p in "${WORKER_PIDS[@]:-}"; do
    kill -TERM "$p" 2>/dev/null || true
  done
  wait || true
  exit 130
}
trap on_interrupt INT TERM

combo_total="$(combo_count)"
echo "[P1-HGR] combo_count=${combo_total} joint_layout_capacity=8 route_profiles=4 combos_per_gpu=${COMBOS_PER_GPU} gpus=${GPU_LIST}"

for ds in "${DATASET_ARR[@]}"; do
  total_jobs=$(( ${#GPUS[@]} * COMBOS_PER_GPU ))
  echo "=== [${ds}] HGR P1 joint fast screen (${total_jobs} runs = ${#GPUS[@]} gpus x ${COMBOS_PER_GPU}) ==="
  if [ "$total_jobs" -lt "$combo_total" ]; then
    echo "[P1-HGR] warning: total_jobs=${total_jobs} < combo_total=${combo_total}; only the first combos will run."
  elif [ "$total_jobs" -gt "$combo_total" ]; then
    echo "[P1-HGR] warning: total_jobs=${total_jobs} > combo_total=${combo_total}; some combos will repeat."
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
        [ -n "$combo_row" ] || { echo "[P1-HGR] missing combo row for idx=${combo_id}" >&2; exit 1; }

        IFS=, read -r arch_tag route_tag layout_id total_layers merge_mode group_mode group_topk moe_topk emb dfeat dexp drouter scale train_bs eval_bs expert_feat macro_scope macro_pool par_temp schedule_preset <<< "$combo_row"
        read -r safe_train_bs safe_eval_bs <<< "$(apply_oom_safety_cap "$ds" "$total_layers" "$emb" "$dexp" "$drouter" "$scale" "$train_bs" "$eval_bs")"

        phase="${PHASE_PREFIX}_C$(printf '%02d' "$combo_id")_${arch_tag}_${route_tag}_${merge_mode}_${group_mode}"
        exp_name="${EXP_NAME_BASE}"
        exp_desc="${EXP_DESC_BASE} combo=C${combo_id} arch=${arch_tag} route=${route_tag} layout=${layout_id} total_layers=${total_layers} merge=${merge_mode} group=${group_mode} expert_feat=${expert_feat} sched=${schedule_preset}"

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
          --expert-use-feature "${expert_feat}"
          --macro-routing-scope "${macro_scope}"
          --macro-session-pooling "${macro_pool}"
          --parallel-stage-gate-temperature "${par_temp}"
          --train-batch-size "${safe_train_bs}"
          --eval-batch-size "${safe_eval_bs}"
          --embedding-size "${emb}"
          --d-feat-emb "${dfeat}"
          --d-expert-hidden "${dexp}"
          --d-router-hidden "${drouter}"
          --expert-scale "${scale}"
          --lr-space "${LR_SPACE}"
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

        echo "[P1-HGR] gpu=${gpu} slot=${slot} combo=C${combo_id} arch=${arch_tag} route=${route_tag} layout=${layout_id}/${total_layers}L merge=${merge_mode} group=${group_mode} bs=${safe_train_bs}/${safe_eval_bs} emb=${emb} feat=${dfeat} exp=${dexp} router=${drouter} scale=${scale} expert_feat=${expert_feat} sched=${schedule_preset}"
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
