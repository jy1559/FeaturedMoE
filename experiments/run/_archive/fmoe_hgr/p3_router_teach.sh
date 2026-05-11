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
COMBOS_PER_GPU="4"
MAX_EVALS="10"
TUNE_EPOCHS="100"
TUNE_PATIENCE="15"
SEED_BASE="1330"
PHASE_PREFIX="P3HGR_router16"
LOG_WANDB="false"
DRY_RUN="${DRY_RUN:-false}"
GROUP_BAL="0.001"
INTRA_BAL="0.001"
EXP_NAME_BASE="P3_hgr_router_teach"
EXP_DESC_BASE="Post-P2 HGR router-teaching phase. Structure stays fixed around the best L15-hybrid anchors, and router supervision/distillation variants are compared."
EXP_FOCUS="arch_layout_id,stage_merge_mode,group_router_mode,group_top_k,embedding_size,d_feat_emb,d_expert_hidden,d_router_hidden,expert_scale,train_batch_size,eval_batch_size,router_distill_enable,router_distill_lambda,router_distill_temperature,router_distill_until,group_feature_spec_aux_lambda,group_feature_spec_stages,learning_rate,weight_decay,hidden_dropout_prob,balance_loss_lambda"

usage() {
  cat <<USAGE
Usage: $0 [--dataset movielens1m|--datasets movielens1m] [--gpus 0,1,2,3]
          [--combos-per-gpu 4] [--max-evals 10] [--tune-epochs 100] [--tune-patience 15]
          [--phase-prefix NAME] [--log-wandb] [--dry-run]
USAGE
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --dataset|--datasets) DATASETS="$2"; shift 2 ;;
    --gpus) GPU_LIST="$2"; shift 2 ;;
    --combos-per-gpu) COMBOS_PER_GPU="$2"; shift 2 ;;
    --max-evals) MAX_EVALS="$2"; shift 2 ;;
    --tune-epochs) TUNE_EPOCHS="$2"; shift 2 ;;
    --tune-patience) TUNE_PATIENCE="$2"; shift 2 ;;
    --seed-base) SEED_BASE="$2"; shift 2 ;;
    --phase-prefix) PHASE_PREFIX="$2"; shift 2 ;;
    --group-balance-lambda) GROUP_BAL="$2"; shift 2 ;;
    --intra-balance-lambda) INTRA_BAL="$2"; shift 2 ;;
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
anchors = [
    {
        "anchor": "A0",
        "layout": 15,
        "merge": "serial",
        "group": "hybrid",
        "group_top_k": 0,
        "moe_top_k": 0,
        "emb": 128,
        "dfeat": 16,
        "dexp": 160,
        "drouter": 64,
        "scale": 3,
        "train_bs": 4096,
        "eval_bs": 8192,
        "base_lr": "1.8e-3",
        "base_wd": "1e-5",
        "lr_space": "9e-4,1.3e-3,1.8e-3,2.3e-3,2.8e-3",
        "wd_space": "1e-6,3e-6,1e-5,2e-5",
    },
    {
        "anchor": "A1",
        "layout": 15,
        "merge": "serial",
        "group": "hybrid",
        "group_top_k": 0,
        "moe_top_k": 0,
        "emb": 160,
        "dfeat": 16,
        "dexp": 256,
        "drouter": 112,
        "scale": 3,
        "train_bs": 2048,
        "eval_bs": 4096,
        "base_lr": "6.5e-4",
        "base_wd": "1.5e-5",
        "lr_space": "3.5e-4,5e-4,6.5e-4,8e-4,1.0e-3",
        "wd_space": "5e-6,1.5e-5,3e-5,5e-5",
    },
]

profiles = [
    {
        "profile": "M0",
        "distill_enable": "false",
        "distill_lambda": "0.0",
        "distill_tau": "1.5",
        "distill_until": "0.2",
        "group_top_k": 0,
        "spec_lambda": "1e-4",
        "spec_stages": "[mid]",
    },
    {
        "profile": "M1",
        "distill_enable": "true",
        "distill_lambda": "2e-3",
        "distill_tau": "1.5",
        "distill_until": "0.2",
        "group_top_k": 0,
        "spec_lambda": "1e-4",
        "spec_stages": "[mid]",
    },
    {
        "profile": "M2",
        "distill_enable": "true",
        "distill_lambda": "5e-3",
        "distill_tau": "1.5",
        "distill_until": "0.2",
        "group_top_k": 0,
        "spec_lambda": "1e-4",
        "spec_stages": "[mid]",
    },
    {
        "profile": "M3",
        "distill_enable": "true",
        "distill_lambda": "1e-2",
        "distill_tau": "1.5",
        "distill_until": "0.2",
        "group_top_k": 0,
        "spec_lambda": "1e-4",
        "spec_stages": "[mid]",
    },
    {
        "profile": "M4",
        "distill_enable": "true",
        "distill_lambda": "5e-3",
        "distill_tau": "1.5",
        "distill_until": "0.3",
        "group_top_k": 0,
        "spec_lambda": "1e-4",
        "spec_stages": "[mid]",
    },
    {
        "profile": "M5",
        "distill_enable": "true",
        "distill_lambda": "5e-3",
        "distill_tau": "1.2",
        "distill_until": "0.2",
        "group_top_k": 0,
        "spec_lambda": "1e-4",
        "spec_stages": "[mid]",
    },
    {
        "profile": "M6",
        "distill_enable": "true",
        "distill_lambda": "5e-3",
        "distill_tau": "1.5",
        "distill_until": "0.2",
        "group_top_k": 0,
        "spec_lambda": "3e-4",
        "spec_stages": "[macro,mid]",
    },
    {
        "profile": "M7",
        "distill_enable": "true",
        "distill_lambda": "5e-3",
        "distill_tau": "1.5",
        "distill_until": "0.2",
        "group_top_k": 2,
        "spec_lambda": "1e-4",
        "spec_stages": "[mid]",
    },
]

drop_space = "0.09,0.10,0.11"
bal_space = "0.0025,0.0032,0.0042"

ordered = [
    (anchors[0], profiles[:4]),
    (anchors[0], profiles[4:]),
    (anchors[1], profiles[:4]),
    (anchors[1], profiles[4:]),
]

for anchor, profs in ordered:
    for prof in profs:
        row = [
            f"{anchor['anchor']}_{prof['profile']}",
            anchor["anchor"],
            prof["profile"],
            anchor["layout"],
            anchor["merge"],
            anchor["group"],
            prof["group_top_k"],
            anchor["moe_top_k"],
            anchor["emb"],
            anchor["dfeat"],
            anchor["dexp"],
            anchor["drouter"],
            anchor["scale"],
            anchor["train_bs"],
            anchor["eval_bs"],
            anchor["base_lr"],
            anchor["base_wd"],
            anchor["lr_space"],
            anchor["wd_space"],
            drop_space,
            bal_space,
            prof["distill_enable"],
            prof["distill_lambda"],
            prof["distill_tau"],
            prof["distill_until"],
            "true",
            prof["spec_lambda"],
            prof["spec_stages"],
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

for ds in "${DATASET_ARR[@]}"; do
  total_jobs=$(( ${#GPUS[@]} * COMBOS_PER_GPU ))
  echo "=== [${ds}] HGR P3 router teaching (${total_jobs} runs = ${#GPUS[@]} gpus x ${COMBOS_PER_GPU}) ==="
  if [ "$total_jobs" -lt "$combo_total" ]; then
    echo "[P3-HGR] warning: total_jobs=${total_jobs} < combo_total=${combo_total}; only the first combos will run."
  elif [ "$total_jobs" -gt "$combo_total" ]; then
    echo "[P3-HGR] warning: total_jobs=${total_jobs} > combo_total=${combo_total}; some combos will repeat."
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
        [ -n "$combo_row" ] || { echo "[P3-HGR] missing combo row for idx=${combo_id}" >&2; exit 1; }

        IFS=$'\t' read -r combo_tag anchor_tag profile_tag layout_id merge_mode group_mode group_topk moe_topk emb dfeat dexp drouter scale train_bs eval_bs base_lr base_wd lr_space wd_space drop_space bal_space distill_enable distill_lambda distill_tau distill_until spec_enable spec_lambda spec_stages <<< "$combo_row"

        phase="${PHASE_PREFIX}_C$(printf '%02d' "$combo_id")_${anchor_tag}_${profile_tag}_${merge_mode}_${group_mode}"
        exp_desc="${EXP_DESC_BASE} combo=${combo_tag} layout=${layout_id} merge=${merge_mode} group=${group_mode} dims=${emb}/${dfeat}/${dexp}/${drouter} scale=${scale} bs=${train_bs}/${eval_bs} distill=${distill_enable}:${distill_lambda}@tau${distill_tau}/until${distill_until} spec=${spec_stages}:${spec_lambda} group_top_k=${group_topk}"

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
          --train-batch-size "${train_bs}"
          --eval-batch-size "${eval_bs}"
          --embedding-size "${emb}"
          --d-feat-emb "${dfeat}"
          --d-expert-hidden "${dexp}"
          --d-router-hidden "${drouter}"
          --expert-scale "${scale}"
          --learning-rate "${base_lr}"
          --weight-decay "${base_wd}"
          --dropout "0.10"
          --balance-loss-lambda "0.0032"
          --group-balance-lambda "${GROUP_BAL}"
          --intra-balance-lambda "${INTRA_BAL}"
          --group-feature-spec-aux-enable "${spec_enable}"
          --group-feature-spec-aux-lambda "${spec_lambda}"
          --group-feature-spec-stages "${spec_stages}"
          --group-feature-spec-min-tokens "8"
          --router-distill-enable "${distill_enable}"
          --router-distill-lambda "${distill_lambda}"
          --router-distill-temperature "${distill_tau}"
          --router-distill-until "${distill_until}"
          --lr-space "${lr_space}"
          --wd-space "${wd_space}"
          --dropout-space "${drop_space}"
          --balance-space "${bal_space}"
          --exp-name "${EXP_NAME_BASE}"
          --exp-desc "${exp_desc}"
          --exp-focus "${EXP_FOCUS}"
          --override "++search.group_balance_lambda=[${GROUP_BAL}]"
          --override "++search.intra_balance_lambda=[${INTRA_BAL}]"
          --override "++search.group_feature_spec_aux_enable=[${spec_enable}]"
          --override "++search.group_feature_spec_aux_lambda=[${spec_lambda}]"
          --override "++search.group_feature_spec_stages=[${spec_stages}]"
          --override "++search.group_feature_spec_min_tokens=[8]"
          --override "++search.router_distill_enable=[${distill_enable}]"
          --override "++search.router_distill_lambda=[${distill_lambda}]"
          --override "++search.router_distill_temperature=[${distill_tau}]"
          --override "++search.router_distill_until=[${distill_until}]"
        )

        if [ "$LOG_WANDB" = "true" ]; then
          cmd+=(--log-wandb)
        else
          cmd+=(--no-wandb)
        fi
        if [ "$DRY_RUN" = "true" ]; then
          cmd+=(--dry-run)
        fi

        echo "[P3-HGR] gpu=${gpu} slot=${slot} combo=${combo_tag} layout=${layout_id} merge=${merge_mode} group=${group_mode} dims=${emb}/${dexp}/${drouter} bs=${train_bs}/${eval_bs} distill=${distill_enable}:${distill_lambda} spec=${spec_stages}:${spec_lambda} gtopk=${group_topk}"
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
