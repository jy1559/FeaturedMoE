#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=/dev/null
source "${RUN_DIR}/common/run_metadata.sh"
# shellcheck source=/dev/null
source "${RUN_DIR}/common/dispatch_gpu_queue.sh"

DATASETS="retail_rocket"
GPU_LIST="0,1,2,3"
COMBOS_PER_GPU=""
MAX_EVALS="12"
TUNE_EPOCHS="30"
TUNE_PATIENCE="6"
SEED_BASE="840"
PHASE_PREFIX="P1RFI"
CATALOG_PROFILE="layout14"
SCHEDULE_PRESET="off"
LOG_WANDB="false"
DRY_RUN="${DRY_RUN:-false}"
OOM_RETRY_MIN_TRAIN_BS="1024"
LR_SPACE="2e-4,3e-4,4e-4,5e-4,6.5e-4,8e-4,1e-3"
DROP_SPACE="0.10"
BAL_SPACE="0.001,0.003"

COMBO_CATALOG=""

usage() {
  cat <<USAGE
Usage: $0 [--datasets retail_rocket] [--gpus 0,1,2,3]
          [--combos-per-gpu N] [--max-evals N] [--tune-epochs N] [--tune-patience N]
          [--combo-catalog spec] [--catalog-profile layout14|recover12|blocked24]
          [--phase-prefix P1RFI] [--seed-base N]
          [--lr-space csv] [--drop-space csv] [--balance-space csv]
          [--oom-retry-min-train-bs N] [--log-wandb] [--dry-run]
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
    --combo-catalog) COMBO_CATALOG="$2"; shift 2 ;;
    --catalog-profile) CATALOG_PROFILE="$2"; shift 2 ;;
    --phase-prefix) PHASE_PREFIX="$2"; shift 2 ;;
    --seed-base) SEED_BASE="$2"; shift 2 ;;
    --lr-space) LR_SPACE="$2"; shift 2 ;;
    --drop-space) DROP_SPACE="$2"; shift 2 ;;
    --balance-space) BAL_SPACE="$2"; shift 2 ;;
    --oom-retry-min-train-bs) OOM_RETRY_MIN_TRAIN_BS="$2"; shift 2 ;;
    --log-wandb) LOG_WANDB="true"; shift ;;
    --no-wandb) LOG_WANDB="false"; shift ;;
    --dry-run) DRY_RUN="true"; shift ;;
    --help|-h) usage; exit 0 ;;
    --) shift; break ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

dispatch_parse_csv "$GPU_LIST" GPUS
[ "${#GPUS[@]}" -eq 0 ] && { echo "Empty GPU list"; exit 1; }

dispatch_parse_csv "$DATASETS" DATASET_ARR
[ "${#DATASET_ARR[@]}" -eq 0 ] && { echo "Empty dataset list"; exit 1; }

if [ -n "$COMBOS_PER_GPU" ] && { ! [[ "$COMBOS_PER_GPU" =~ ^[0-9]+$ ]] || [ "$COMBOS_PER_GPU" -le 0 ]; }; then
  echo "--combos-per-gpu must be a positive integer" >&2
  exit 1
fi
if ! [[ "$MAX_EVALS" =~ ^[0-9]+$ ]] || [ "$MAX_EVALS" -le 0 ]; then
  echo "--max-evals must be a positive integer" >&2
  exit 1
fi
if ! [[ "$TUNE_EPOCHS" =~ ^[0-9]+$ ]] || [ "$TUNE_EPOCHS" -le 0 ]; then
  echo "--tune-epochs must be a positive integer" >&2
  exit 1
fi
if ! [[ "$TUNE_PATIENCE" =~ ^[0-9]+$ ]] || [ "$TUNE_PATIENCE" -le 0 ]; then
  echo "--tune-patience must be a positive integer" >&2
  exit 1
fi

build_blocked24_combo_catalog() {
  python3 - <<'PY'
rows = [
    # 12 shared anchors: same 3 combos repeated across 4 serial layouts
    (16, "serial",   128, 16, 128,  64, 3, 4096, 8192),
    (16, "serial",   160, 16, 160,  80, 3, 4096, 8192),
    (16, "serial",   192, 24, 192,  96, 3, 3072, 6144),
    (18, "serial",   128, 16, 128,  64, 3, 4096, 8192),
    (18, "serial",   160, 16, 160,  80, 3, 4096, 8192),
    (18, "serial",   192, 24, 192,  96, 3, 3072, 6144),
    (15, "serial",   128, 16, 128,  64, 3, 4096, 8192),
    (15, "serial",   160, 16, 160,  80, 3, 4096, 8192),
    (15, "serial",   192, 24, 192,  96, 3, 3072, 6144),
    (7,  "serial",   128, 16, 128,  64, 3, 4096, 8192),
    (7,  "serial",   160, 16, 160,  80, 3, 4096, 8192),
    (7,  "serial",   192, 24, 192,  96, 3, 3072, 6144),
    # 8 dim probes: only top prior layouts get additional capacity sweeps
    (16, "serial",   128, 24, 160,  64, 3, 4096, 8192),
    (16, "serial",   160, 24, 192,  96, 3, 3072, 6144),
    (16, "serial",   224, 16, 256, 112, 2, 1536, 3072),
    (16, "serial",   224, 24, 320, 128, 2, 1024, 2048),
    (18, "serial",   128, 24, 160,  64, 3, 4096, 8192),
    (18, "serial",   160, 24, 192,  96, 3, 3072, 6144),
    (18, "serial",   224, 16, 256, 112, 2, 1536, 3072),
    (18, "serial",   224, 24, 320, 128, 2, 1024, 2048),
    # 4 parallel sentinels: same base/medium combos to contrast execution family
    (21, "parallel", 128, 16, 128,  64, 3, 3072, 6144),
    (21, "parallel", 160, 16, 160,  80, 3, 3072, 6144),
    (26, "parallel", 128, 16, 128,  64, 3, 3072, 6144),
    (26, "parallel", 160, 16, 160,  80, 3, 3072, 6144),
]
print(";".join(",".join(str(x) for x in row) for row in rows))
PY
}

build_recover12_combo_catalog() {
  python3 - <<'PY'
rows = [
    # 12 focused probes: keep only top-performing serial layouts and medium-capacity combos.
    (16, "serial", 128, 16, 128,  64, 3, 4096, 8192),
    (16, "serial", 160, 16, 160,  80, 3, 4096, 8192),
    (16, "serial", 128, 24, 160,  64, 3, 4096, 8192),
    (16, "serial", 160, 24, 192,  96, 3, 3072, 6144),
    (18, "serial", 128, 16, 128,  64, 3, 4096, 8192),
    (18, "serial", 160, 16, 160,  80, 3, 4096, 8192),
    (18, "serial", 128, 24, 160,  64, 3, 4096, 8192),
    (18, "serial", 160, 24, 192,  96, 3, 3072, 6144),
    (15, "serial", 128, 16, 128,  64, 3, 4096, 8192),
    (15, "serial", 160, 16, 160,  80, 3, 4096, 8192),
    (15, "serial", 128, 24, 160,  64, 3, 4096, 8192),
    (15, "serial", 160, 24, 192,  96, 3, 3072, 6144),
]
print(";".join(",".join(str(x) for x in row) for row in rows))
PY
}

build_layout14_combo_catalog() {
  python3 - <<'PY'
rows = [
    # 10 structure-first anchors: compare top serial layouts under 2 shared dims
    (16, "serial", 128, 16, 128,  64, 3, 4096, 8192),
    (18, "serial", 128, 16, 128,  64, 3, 4096, 8192),
    (15, "serial", 128, 16, 128,  64, 3, 4096, 8192),
    (7,  "serial", 128, 16, 128,  64, 3, 4096, 8192),
    (5,  "serial", 128, 16, 128,  64, 3, 4096, 8192),
    (16, "serial", 160, 16, 160,  80, 3, 4096, 8192),
    (18, "serial", 160, 16, 160,  80, 3, 4096, 8192),
    (15, "serial", 160, 16, 160,  80, 3, 4096, 8192),
    (7,  "serial", 160, 16, 160,  80, 3, 4096, 8192),
    (5,  "serial", 160, 16, 160,  80, 3, 4096, 8192),
    # 4 light dim probes: only on prior top layouts
    (16, "serial", 128, 24, 160,  64, 3, 4096, 8192),
    (18, "serial", 128, 24, 160,  64, 3, 4096, 8192),
    (16, "serial", 160, 24, 192,  96, 3, 3072, 6144),
    (18, "serial", 160, 24, 192,  96, 3, 3072, 6144),
]
print(";".join(",".join(str(x) for x in row) for row in rows))
PY
}

build_probe_combo_catalog() {
  case "${CATALOG_PROFILE}" in
    layout14)
      build_layout14_combo_catalog
      ;;
    blocked24)
      build_blocked24_combo_catalog
      ;;
    recover12)
      build_recover12_combo_catalog
      ;;
    *)
      echo "Unsupported --catalog-profile=${CATALOG_PROFILE}" >&2
      exit 1
      ;;
  esac
}

read_combo() {
  local combo_catalog="$1"
  local idx="$2"
  python3 - <<'PY' "$combo_catalog" "$idx"
import sys

raw = sys.argv[1]
idx = int(sys.argv[2])
rows = [x.strip() for x in raw.split(';') if x.strip()]
if not rows:
    raise SystemExit("empty combo catalog")
row = rows[idx % len(rows)]
parts = [x.strip() for x in row.split(',')]
if len(parts) != 9:
    raise SystemExit(f"invalid combo row: {row}")
print(" ".join(parts))
PY
}

combo_count() {
  local combo_catalog="$1"
  python3 - <<'PY' "$combo_catalog"
import sys

rows = [x.strip() for x in sys.argv[1].split(';') if x.strip()]
print(len(rows))
PY
}

compute_auto_combos_per_gpu() {
  local combo_n="$1"
  local gpu_n="$2"
  echo $(( (combo_n + gpu_n - 1) / gpu_n ))
}

compute_covered_combos() {
  local combo_n="$1"
  local gpu_n="$2"
  local combos_per_gpu="$3"
  local total_slots=$(( gpu_n * combos_per_gpu ))

  if [ "$total_slots" -gt "$combo_n" ]; then
    echo "$combo_n"
  else
    echo "$total_slots"
  fi
}

planned_combo_count_for_gpu() {
  local gpu_idx="$1"
  local covered_combo_n="$2"
  local gpu_n="$3"
  local combos_per_gpu="$4"
  local count=0
  local slot combo_idx

  for ((slot=0; slot<combos_per_gpu; slot++)); do
    combo_idx=$(( slot * gpu_n + gpu_idx ))
    if [ "$combo_idx" -lt "$covered_combo_n" ]; then
      count=$((count + 1))
    fi
  done

  echo "$count"
}

lr_space_for_combo() {
  local execution="$1"
  local emb="$2"
  local dfeat="$3"
  local dexp="$4"
  local drouter="$5"

  if [ -n "$LR_SPACE" ]; then
    echo "$LR_SPACE"
    return 0
  fi

  if [ "$emb" -ge 224 ] || [ "$dexp" -ge 320 ] || [ "$drouter" -ge 128 ]; then
    echo "1.5e-4,2.2e-4,3e-4,4e-4,5.2e-4,6.5e-4"
  elif [ "$execution" = "parallel" ] || [ "$emb" -ge 160 ] || [ "$dexp" -ge 192 ] || [ "$drouter" -ge 96 ] || [ "$dfeat" -ge 24 ]; then
    echo "2e-4,3e-4,4.2e-4,5.5e-4,7e-4,8.5e-4"
  else
    echo "2.5e-4,3.5e-4,4.5e-4,5.8e-4,7.2e-4,9e-4"
  fi
}

wd_space_for_combo() {
  local execution="$1"
  local emb="$2"
  local dexp="$3"
  local drouter="$4"

  if [ "${CATALOG_PROFILE}" = "recover12" ] || [ "${CATALOG_PROFILE}" = "layout14" ]; then
    echo "0,1e-5,2.5e-5,5e-5,7.5e-5,1e-4"
    return 0
  fi

  if [ "$execution" = "parallel" ] || [ "$emb" -ge 224 ] || [ "$dexp" -ge 320 ] || [ "$drouter" -ge 128 ]; then
    echo "0,5e-6,1e-5,2.5e-5,5e-5"
  else
    echo "0,5e-6,1e-5,2.5e-5,5e-5,1e-4"
  fi
}

apply_rr_oom_cap() {
  local emb="$1"
  local dfeat="$2"
  local dexp="$3"
  local drouter="$4"
  local expert_scale="$5"
  local train_bs="$6"
  local eval_bs="$7"

  local capped_train="$train_bs"
  local capped_eval="$eval_bs"

  if [ "$emb" -ge 256 ] || [ "$dexp" -ge 384 ] || [ "$drouter" -ge 128 ]; then
    if [ "$capped_train" -gt 1024 ]; then
      capped_train=1024
      capped_eval=2048
    fi
  elif [ "$emb" -ge 224 ] || [ "$dexp" -ge 320 ] || [ "$drouter" -ge 112 ] || [ "$expert_scale" -ge 4 ] || [ "$dfeat" -ge 32 ]; then
    if [ "$capped_train" -gt 1536 ]; then
      capped_train=1536
      capped_eval=3072
    fi
  elif [ "$emb" -ge 192 ] || [ "$dexp" -ge 224 ] || [ "$drouter" -ge 96 ]; then
    if [ "$capped_train" -gt 3072 ]; then
      capped_train=3072
      capped_eval=6144
    fi
  fi

  echo "${capped_train} ${capped_eval}"
}

latest_phase_log() {
  local dataset="$1"
  local phase="$2"
  local dataset_tag model_tag phase_bucket dir
  dataset_tag="$(run_dataset_tag "$dataset")"
  model_tag="$(run_model_tag "FeaturedMoE_v2_HiR")"
  phase_bucket="$(run_sanitize "${phase%%_*}")"
  dir="$(run_log_dir fmoe_v2_hir)/hparam/${phase_bucket}/${dataset_tag}/${model_tag}"
  [ -d "$dir" ] || { echo ""; return 0; }
  ls -1t "$dir"/*"_hparam_${phase}.log" 2>/dev/null | head -n1
}

is_oom_log() {
  local log_file="$1"
  [ -n "$log_file" ] || return 1
  [ -f "$log_file" ] || return 1
  grep -qi "CUDA out of memory" "$log_file"
}

if [ -z "$COMBO_CATALOG" ]; then
  COMBO_CATALOG="$(build_probe_combo_catalog)"
fi

COMBO_N="$(combo_count "$COMBO_CATALOG")"
[ "$COMBO_N" -le 0 ] && { echo "Probe combo catalog is empty"; exit 1; }

if [ -z "$COMBOS_PER_GPU" ]; then
  COMBOS_PER_GPU="$(compute_auto_combos_per_gpu "$COMBO_N" "${#GPUS[@]}")"
fi
[ "$COMBOS_PER_GPU" -le 0 ] && { echo "invalid combos-per-gpu"; exit 1; }
COVERED_COMBOS="$(compute_covered_combos "$COMBO_N" "${#GPUS[@]}" "$COMBOS_PER_GPU")"

EXP_DIR="$(run_experiments_dir)"
cd "$EXP_DIR"
run_export_runtime_env

INTERRUPTED="false"
WORKER_PIDS=()
on_interrupt() {
  INTERRUPTED="true"
  echo "[INTERRUPT] stopping RR factorized P1 probe workers..."
  local p
  for p in "${WORKER_PIDS[@]:-}"; do
    kill -TERM "$p" 2>/dev/null || true
  done
  wait || true
  exit 130
}
trap on_interrupt INT TERM

for ds in "${DATASET_ARR[@]}"; do
  echo "=== [${ds}] P1 factorized RR probe combos=${COVERED_COMBOS}/${COMBO_N} gpus=${#GPUS[@]} cpg=${COMBOS_PER_GPU} ==="
  gpu_plan=()
  for gidx in "${!GPUS[@]}"; do
    gpu="${GPUS[$gidx]}"
    planned_count="$(planned_combo_count_for_gpu "$gidx" "$COVERED_COMBOS" "${#GPUS[@]}" "$COMBOS_PER_GPU")"
    gpu_plan+=("G${gpu}=${planned_count}")
  done
  echo "[${PHASE_PREFIX}] gpu_combo_plan=$(IFS=,; echo "${gpu_plan[*]}")"

  WORKER_PIDS=()
  for gidx in "${!GPUS[@]}"; do
    gpu="${GPUS[$gidx]}"
    (
      set -euo pipefail
      slot=0
      combo_idx=0
      for ((slot=0; slot<COMBOS_PER_GPU; slot++)); do
        combo_idx=$(( slot * ${#GPUS[@]} + gidx ))
        if [ "$combo_idx" -ge "$COVERED_COMBOS" ]; then
          continue
        fi

        read -r layout_id execution emb dfeat dexp drouter expert_scale train_bs eval_bs <<< "$(read_combo "$COMBO_CATALOG" "$combo_idx")"
        read -r train_bs eval_bs <<< "$(apply_rr_oom_cap "$emb" "$dfeat" "$dexp" "$drouter" "$expert_scale" "$train_bs" "$eval_bs")"
        lr_space="$(lr_space_for_combo "$execution" "$emb" "$dfeat" "$dexp" "$drouter")"
        wd_space="$(wd_space_for_combo "$execution" "$emb" "$dexp" "$drouter")"
        seed=$(( SEED_BASE + combo_idx ))
        phase="${PHASE_PREFIX}_G${gpu}_C$(printf '%02d' "$combo_idx")_${execution}_L${layout_id}_E${emb}_F${dfeat}_H${dexp}_R${drouter}_B${train_bs}"
        exp_name="${PHASE_PREFIX}_${ds}_${execution}_layout${layout_id}"
        if [ "${CATALOG_PROFILE}" = "layout14" ]; then
          exp_desc="${PHASE_PREFIX}: RR factorized-router layout-first P1. Compare 5 strong serial layouts under 2 shared core dims, then add 2 light dim probes only on L16/L18 to check weak layout-dim interaction before P2."
        elif [ "${CATALOG_PROFILE}" = "recover12" ]; then
          exp_desc="${PHASE_PREFIX}: RR factorized-router recovery P1. Focus L16/L18/L15 with 4 medium-capacity combos each; tighten LR/WD around the viable band before narrow P2."
        else
          exp_desc="${PHASE_PREFIX}: RR factorized-router blocked-joint P1. 12 shared anchors for layout comparison + 8 top-layout dim probes + 4 parallel sentinels; map LR band before narrow P2."
        fi
        exp_focus="fmoe_stage_execution_mode,fmoe_v2_layout_id,embedding_size,d_feat_emb,d_expert_hidden,d_router_hidden,expert_scale,train_batch_size,eval_batch_size,hidden_dropout_prob,balance_loss_lambda,learning_rate,weight_decay"

        echo "[${PHASE_PREFIX}][${ds}] gpu=${gpu} combo=${combo_idx}/${COMBO_N} exec=${execution} layout=L${layout_id} emb=${emb} dfeat=${dfeat} dexp=${dexp} drouter=${drouter} scale=${expert_scale} bs=${train_bs}/${eval_bs}"

        cmd=(
          bash "${SCRIPT_DIR}/tune_hparam.sh"
          --dataset "$ds"
          --layout-id "$layout_id"
          --execution "$execution"
          --schedule-preset "$SCHEDULE_PRESET"
          --gpu "$gpu"
          --max-evals "$MAX_EVALS"
          --tune-epochs "$TUNE_EPOCHS"
          --tune-patience "$TUNE_PATIENCE"
          --seed "$seed"
          --phase "$phase"
          --search-profile p1_shallow
          --train-batch-size "$train_bs"
          --eval-batch-size "$eval_bs"
          --embedding-size "$emb"
          --num-heads 8
          --d-feat-emb "$dfeat"
          --d-expert-hidden "$dexp"
          --d-router-hidden "$drouter"
          --expert-scale "$expert_scale"
          --lr-space "$lr_space"
          --wd-space "$wd_space"
          --dropout-space "$DROP_SPACE"
          --balance-space "$BAL_SPACE"
          --exp-name "$exp_name"
          --exp-desc "$exp_desc"
          --exp-focus "$exp_focus"
        )
        if [ "$LOG_WANDB" = "true" ]; then
          cmd+=(--log-wandb)
        else
          cmd+=(--no-wandb)
        fi
        if [ "$DRY_RUN" = "true" ]; then
          cmd+=(--dry-run)
        fi

        set +e
        "${cmd[@]}"
        rc=$?
        set -e

        if [ "$rc" -ne 0 ]; then
          log_path="$(latest_phase_log "$ds" "$phase")"
          if is_oom_log "$log_path" && [ "$train_bs" -ge $((OOM_RETRY_MIN_TRAIN_BS * 2)) ]; then
            retry_train_bs=$(( train_bs / 2 ))
            retry_eval_bs=$(( eval_bs / 2 ))
            retry_phase="${phase}_RBS${retry_train_bs}"
            retry_desc="${exp_desc} OOM-retry(train/eval=${retry_train_bs}/${retry_eval_bs})."
            echo "[${PHASE_PREFIX}][OOM-RETRY] ${ds} ${execution} L${layout_id} phase=${phase} -> ${retry_phase} bs ${train_bs}/${eval_bs} -> ${retry_train_bs}/${retry_eval_bs}"
            retry_cmd=(
              bash "${SCRIPT_DIR}/tune_hparam.sh"
              --dataset "$ds"
              --layout-id "$layout_id"
              --execution "$execution"
              --schedule-preset "$SCHEDULE_PRESET"
              --gpu "$gpu"
              --max-evals "$MAX_EVALS"
              --tune-epochs "$TUNE_EPOCHS"
              --tune-patience "$TUNE_PATIENCE"
              --seed "$seed"
              --phase "$retry_phase"
              --search-profile p1_shallow
              --train-batch-size "$retry_train_bs"
              --eval-batch-size "$retry_eval_bs"
              --embedding-size "$emb"
              --num-heads 8
              --d-feat-emb "$dfeat"
              --d-expert-hidden "$dexp"
              --d-router-hidden "$drouter"
              --expert-scale "$expert_scale"
              --lr-space "$lr_space"
              --wd-space "$wd_space"
              --dropout-space "$DROP_SPACE"
              --balance-space "$BAL_SPACE"
              --exp-name "$exp_name"
              --exp-desc "$retry_desc"
              --exp-focus "$exp_focus"
            )
            if [ "$LOG_WANDB" = "true" ]; then
              retry_cmd+=(--log-wandb)
            else
              retry_cmd+=(--no-wandb)
            fi
            if [ "$DRY_RUN" = "true" ]; then
              retry_cmd+=(--dry-run)
            fi
            "${retry_cmd[@]}"
          else
            exit "$rc"
          fi
        fi
      done
    ) &
    WORKER_PIDS+=("$!")
  done

  FAIL=0
  for p in "${WORKER_PIDS[@]}"; do
    if ! wait "$p"; then
      FAIL=1
    fi
  done

  if [ "$FAIL" -ne 0 ]; then
    echo "[ERROR] RR factorized P1 probe failed for dataset=${ds}" >&2
    exit 1
  fi

  echo "=== [${ds}] P1 factorized RR probe done ==="
done

trap - INT TERM

if [ "$INTERRUPTED" = "true" ]; then
  exit 130
fi

if [ "$DRY_RUN" != "true" ]; then
  run_update_track_report fmoe_v2_hir
fi
