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
LAYOUT_IDS="16,18,7"
SURPRISE_LAYOUT_IDS="5,15"
COMBOS_PER_GPU=""
SURPRISE_COMBOS_PER_GPU=""
MAX_EVALS="15"
TUNE_EPOCHS="50"
TUNE_PATIENCE="10"
SEED_BASE="420"
PHASE_PREFIX="P2RRF"
SURPRISE_PHASE_PREFIX=""
SCHEDULE_PRESET="off"
LOG_WANDB="false"
DRY_RUN="${DRY_RUN:-false}"
OOM_RETRY_MIN_TRAIN_BS="1024"
DROP_SPACE="0.08,0.10,0.12"
BAL_SPACE="0.003,0.006,0.01,0.02"

COMBO_CATALOG=""
SURPRISE_COMBO_CATALOG=""

usage() {
  cat <<USAGE
Usage: $0 [--datasets retail_rocket] [--gpus 0,1,2,3]
          [--layout-ids 16,18,7] [--surprise-layout-ids 5,15]
          [--combos-per-gpu N] [--surprise-combos-per-gpu N]
          [--max-evals N] [--tune-epochs N] [--tune-patience N]
          [--combo-catalog spec] [--surprise-combo-catalog spec]
          [--phase-prefix P2RRF] [--surprise-phase-prefix P2RRFX] [--seed-base N]
          [--drop-space csv] [--balance-space csv]
          [--oom-retry-min-train-bs N] [--log-wandb] [--dry-run]
USAGE
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --datasets) DATASETS="$2"; shift 2 ;;
    --gpus) GPU_LIST="$2"; shift 2 ;;
    --layout-ids) LAYOUT_IDS="$2"; shift 2 ;;
    --surprise-layout-ids) SURPRISE_LAYOUT_IDS="$2"; shift 2 ;;
    --no-surprise-layouts) SURPRISE_LAYOUT_IDS=""; shift ;;
    --combos-per-gpu) COMBOS_PER_GPU="$2"; shift 2 ;;
    --surprise-combos-per-gpu) SURPRISE_COMBOS_PER_GPU="$2"; shift 2 ;;
    --max-evals) MAX_EVALS="$2"; shift 2 ;;
    --tune-epochs) TUNE_EPOCHS="$2"; shift 2 ;;
    --tune-patience) TUNE_PATIENCE="$2"; shift 2 ;;
    --combo-catalog) COMBO_CATALOG="$2"; shift 2 ;;
    --surprise-combo-catalog) SURPRISE_COMBO_CATALOG="$2"; shift 2 ;;
    --phase-prefix) PHASE_PREFIX="$2"; shift 2 ;;
    --surprise-phase-prefix) SURPRISE_PHASE_PREFIX="$2"; shift 2 ;;
    --seed-base) SEED_BASE="$2"; shift 2 ;;
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

[ -z "$SURPRISE_PHASE_PREFIX" ] && SURPRISE_PHASE_PREFIX="${PHASE_PREFIX}X"

dispatch_parse_csv "$GPU_LIST" GPUS
[ "${#GPUS[@]}" -eq 0 ] && { echo "Empty GPU list"; exit 1; }

dispatch_parse_csv "$DATASETS" DATASET_ARR
[ "${#DATASET_ARR[@]}" -eq 0 ] && { echo "Empty dataset list"; exit 1; }

dispatch_parse_csv "$LAYOUT_IDS" LAYOUT_ARR
[ "${#LAYOUT_ARR[@]}" -eq 0 ] && { echo "Empty layout list"; exit 1; }

dispatch_parse_csv "$SURPRISE_LAYOUT_IDS" SURPRISE_LAYOUT_ARR

if [ -n "$COMBOS_PER_GPU" ] && { ! [[ "$COMBOS_PER_GPU" =~ ^[0-9]+$ ]] || [ "$COMBOS_PER_GPU" -le 0 ]; }; then
  echo "--combos-per-gpu must be a positive integer" >&2
  exit 1
fi
if [ -n "$SURPRISE_COMBOS_PER_GPU" ] && { ! [[ "$SURPRISE_COMBOS_PER_GPU" =~ ^[0-9]+$ ]] || [ "$SURPRISE_COMBOS_PER_GPU" -le 0 ]; }; then
  echo "--surprise-combos-per-gpu must be a positive integer" >&2
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

build_main_combo_catalog() {
  python3 - <<'PY'
rows = [
    (128,16,128,64,3,4096,8192), (128,16,128,64,3,3072,6144), (128,16,160,64,3,4096,8192), (128,16,192,80,3,4096,8192), (128,24,160,64,3,4096,8192),
    (160,16,160,80,3,4096,8192), (160,16,192,80,3,4096,8192), (160,24,192,96,3,4096,8192), (160,24,256,96,3,4096,8192), (192,16,192,96,3,4096,8192),
    (192,24,224,96,3,4096,8192), (192,24,256,96,3,4096,8192), (192,24,320,112,3,3072,6144), (224,16,256,112,3,3072,6144), (224,24,320,128,2,3072,6144),
    (224,24,384,128,2,2048,4096), (256,16,320,128,2,2048,4096), (256,24,384,128,2,2048,4096), (96,32,192,192,2,2048,4096), (192,32,256,128,2,3072,6144),
    (96,24,128,96,2,4096,8192), (96,32,160,160,2,3072,6144), (128,24,192,96,3,3072,6144), (128,32,224,128,2,3072,6144), (128,32,256,160,2,2048,4096),
    (160,24,160,112,3,4096,8192), (160,32,224,128,2,3072,6144), (160,32,320,160,2,2048,4096), (192,16,288,96,2,3072,6144), (192,32,320,160,2,2048,4096),
    (192,40,256,192,2,2048,4096), (224,16,320,112,2,3072,6144), (224,32,320,160,2,2048,4096), (224,32,448,160,2,1536,3072), (256,16,448,128,2,2048,4096),
    (256,32,448,160,2,1536,3072), (288,16,384,128,1,1536,3072), (288,24,512,160,1,1024,2048), (320,16,384,160,1,1024,2048), (160,48,192,224,2,1536,3072),
    (128,16,160,80,4,2048,4096), (128,24,192,96,4,1536,3072), (160,16,192,96,4,2048,4096), (160,24,256,112,4,1536,3072), (192,16,256,112,4,2048,4096),
    (192,24,320,128,4,1536,3072), (224,16,320,128,3,2048,4096), (224,24,384,144,3,1536,3072), (224,32,512,160,2,1024,2048), (256,16,512,128,2,1024,2048),
    (256,24,576,160,1,1024,2048), (288,16,448,144,1,1024,2048), (288,32,512,192,1,768,1536), (320,24,512,160,1,768,1536), (320,32,576,192,1,512,1024),
    (96,48,224,224,2,1024,2048), (128,48,256,192,2,1024,2048), (160,40,320,192,2,1024,2048), (192,48,320,224,1,768,1536), (224,48,384,224,1,512,1024),
    (96,16,96,64,4,6144,12288), (96,24,128,80,4,4096,8192), (128,16,96,96,4,4096,8192), (128,24,128,112,4,3072,6144), (160,16,128,128,3,3072,6144),
    (160,24,160,144,3,3072,6144), (160,32,192,160,3,2048,4096), (192,16,160,144,3,3072,6144), (192,24,192,160,3,3072,6144), (192,32,224,192,2,2048,4096),
    (224,16,192,160,3,3072,6144), (224,24,224,176,2,2048,4096), (224,32,256,192,2,2048,4096), (256,16,256,160,2,2048,4096), (256,24,320,176,2,1536,3072),
    (288,16,320,192,1,1024,2048), (288,32,384,224,1,768,1536), (320,16,448,192,1,768,1536), (192,16,384,64,1,3072,6144), (160,48,160,96,4,2048,4096),
]
if len(rows) != 80:
    raise SystemExit(f"expected 80 rows, got {len(rows)}")
print(";".join(",".join(str(x) for x in row) for row in rows))
PY
}

build_surprise_combo_catalog() {
  python3 - <<'PY'
rows = [
    (128,16,128,64,3,4096,8192), (160,16,160,80,3,4096,8192), (192,24,224,96,3,4096,8192), (224,16,256,112,2,3072,6144),
    (96,32,192,192,2,2048,4096), (160,32,224,128,2,3072,6144), (128,48,256,192,2,1024,2048), (256,24,576,160,1,1024,2048),
    (192,16,384,64,1,3072,6144), (224,32,448,160,2,1536,3072), (96,24,128,80,4,4096,8192), (320,16,384,160,1,1024,2048),
    (160,24,160,144,3,3072,6144), (192,48,320,224,1,768,1536), (224,48,384,224,1,512,1024), (288,32,384,224,1,768,1536),
]
if len(rows) != 16:
    raise SystemExit(f"expected 16 rows, got {len(rows)}")
print(";".join(",".join(str(x) for x in row) for row in rows))
PY
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
if len(parts) != 7:
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
  local emb="$1"
  local dfeat="$2"
  local dexp="$3"
  local drouter="$4"

  if [ "$emb" -ge 288 ] || [ "$dexp" -ge 512 ] || [ "$drouter" -ge 160 ] || [ "$dfeat" -ge 48 ]; then
    echo "1.5e-4,2e-4,3e-4,4e-4,5e-4,6.5e-4,8e-4"
  elif [ "$emb" -ge 224 ] || [ "$dexp" -ge 384 ] || [ "$drouter" -ge 128 ] || [ "$dfeat" -ge 32 ]; then
    echo "2e-4,3e-4,4e-4,5e-4,6.5e-4,8e-4,1e-3"
  elif [ "$emb" -ge 192 ] || [ "$dexp" -ge 256 ] || [ "$drouter" -ge 96 ]; then
    echo "2.5e-4,3.5e-4,5e-4,6.5e-4,8e-4,9.5e-4,1.15e-3"
  else
    echo "3e-4,4.5e-4,6e-4,7.5e-4,9e-4,1.05e-3,1.2e-3"
  fi
}

wd_space_for_combo() {
  local emb="$1"
  local dfeat="$2"
  local dexp="$3"
  local drouter="$4"

  if [ "$emb" -ge 288 ] || [ "$dexp" -ge 512 ] || [ "$drouter" -ge 160 ] || [ "$dfeat" -ge 48 ]; then
    echo "0,1e-6,5e-6,1e-5,2.5e-5,5e-5"
  elif [ "$emb" -ge 224 ] || [ "$dexp" -ge 384 ] || [ "$drouter" -ge 128 ] || [ "$dfeat" -ge 32 ]; then
    echo "0,1e-6,5e-6,1e-5,2.5e-5,5e-5"
  elif [ "$emb" -ge 192 ] || [ "$dexp" -ge 256 ] || [ "$drouter" -ge 96 ]; then
    echo "0,5e-6,1e-5,2.5e-5,5e-5,1e-4"
  else
    echo "0,1e-5,2.5e-5,5e-5,1e-4,1.5e-4"
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

  if [ "$emb" -ge 320 ] || [ "$dexp" -ge 576 ] || [ "$drouter" -ge 192 ] || [ "$dfeat" -ge 48 ]; then
    if [ "$capped_train" -gt 1024 ]; then
      capped_train=1024
      capped_eval=2048
    fi
  elif [ "$emb" -ge 288 ] || [ "$dexp" -ge 512 ] || [ "$drouter" -ge 160 ] || [ "$expert_scale" -ge 4 ]; then
    if [ "$capped_train" -gt 1536 ]; then
      capped_train=1536
      capped_eval=3072
    fi
  elif [ "$emb" -ge 256 ] || [ "$dexp" -ge 448 ] || [ "$drouter" -ge 128 ]; then
    if [ "$capped_train" -gt 2048 ]; then
      capped_train=2048
      capped_eval=4096
    fi
  elif [ "$emb" -ge 224 ] || [ "$dexp" -ge 384 ] || [ "$drouter" -ge 112 ] || [ "$dfeat" -ge 32 ]; then
    if [ "$capped_train" -gt 3072 ]; then
      capped_train=3072
      capped_eval=6144
    fi
  elif [ "$dexp" -ge 256 ] || [ "$drouter" -ge 96 ]; then
    if [ "$capped_train" -gt 4096 ]; then
      capped_train=4096
      capped_eval=8192
    fi
  fi

  echo "${capped_train} ${capped_eval}"
}

latest_phase_log() {
  local dataset="$1"
  local phase="$2"
  local dataset_tag model_tag phase_bucket dir
  dataset_tag="$(run_dataset_tag "$dataset")"
  model_tag="$(run_model_tag "FeaturedMoE_v2")"
  phase_bucket="$(run_sanitize "${phase%%_*}")"
  dir="$(run_log_dir fmoe_v2)/hparam/${phase_bucket}/${dataset_tag}/${model_tag}"
  [ -d "$dir" ] || { echo ""; return 0; }
  ls -1t "$dir"/*"_hparam_${phase}.log" 2>/dev/null | head -n1
}

is_oom_log() {
  local log_file="$1"
  [ -n "$log_file" ] || return 1
  [ -f "$log_file" ] || return 1
  grep -qi "CUDA out of memory" "$log_file"
}

find_best_parent_result() {
  local dataset="$1"
  local layout_id="$2"
  python3 - <<'PY' "$(run_results_dir fmoe_v2)" "$dataset" "$layout_id"
import glob
import json
import os
import sys

base = sys.argv[1]
dataset = sys.argv[2]
layout = int(sys.argv[3])
pattern = os.path.join(base, f"{dataset}_FeaturedMoE_v2_p1s_*_l{layout}_*.json")
best_path = ""
best_mrr = None
for fp in glob.glob(pattern):
    try:
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        continue
    mrr = data.get("best_mrr@20")
    if not isinstance(mrr, (int, float)):
        continue
    if best_mrr is None or mrr > best_mrr:
        best_mrr = float(mrr)
        best_path = fp
print(best_path)
PY
}

if [ -z "$COMBO_CATALOG" ]; then
  COMBO_CATALOG="$(build_main_combo_catalog)"
fi
if [ -z "$SURPRISE_COMBO_CATALOG" ]; then
  SURPRISE_COMBO_CATALOG="$(build_surprise_combo_catalog)"
fi

MAIN_COMBO_N="$(combo_count "$COMBO_CATALOG")"
[ "$MAIN_COMBO_N" -le 0 ] && { echo "Main combo catalog is empty"; exit 1; }

if [ -z "$COMBOS_PER_GPU" ]; then
  COMBOS_PER_GPU="$(compute_auto_combos_per_gpu "$MAIN_COMBO_N" "${#GPUS[@]}")"
fi
[ "$COMBOS_PER_GPU" -le 0 ] && { echo "invalid combos-per-gpu"; exit 1; }
MAIN_COVERED_COMBOS="$(compute_covered_combos "$MAIN_COMBO_N" "${#GPUS[@]}" "$COMBOS_PER_GPU")"

SURPRISE_COMBO_N="0"
SURPRISE_COVERED_COMBOS="0"
if [ "${#SURPRISE_LAYOUT_ARR[@]}" -gt 0 ]; then
  SURPRISE_COMBO_N="$(combo_count "$SURPRISE_COMBO_CATALOG")"
  [ "$SURPRISE_COMBO_N" -le 0 ] && { echo "Surprise combo catalog is empty"; exit 1; }
  if [ -z "$SURPRISE_COMBOS_PER_GPU" ]; then
    SURPRISE_COMBOS_PER_GPU="$(compute_auto_combos_per_gpu "$SURPRISE_COMBO_N" "${#GPUS[@]}")"
  fi
  [ "$SURPRISE_COMBOS_PER_GPU" -le 0 ] && { echo "invalid surprise-combos-per-gpu"; exit 1; }
  SURPRISE_COVERED_COMBOS="$(compute_covered_combos "$SURPRISE_COMBO_N" "${#GPUS[@]}" "$SURPRISE_COMBOS_PER_GPU")"
fi

EXP_DIR="$(run_experiments_dir)"
cd "$EXP_DIR"
run_export_runtime_env

INTERRUPTED="false"
WORKER_PIDS=()
on_interrupt() {
  INTERRUPTED="true"
  echo "[INTERRUPT] stopping RR focus P2 workers..."
  local p
  for p in "${WORKER_PIDS[@]:-}"; do
    kill -TERM "$p" 2>/dev/null || true
  done
  wait || true
  exit 130
}
trap on_interrupt INT TERM

run_layout_jobs_for_dataset() {
  local ds="$1"; shift
  local layout_idx="$1"; shift
  local layout_id="$1"; shift
  local parent_result="$1"; shift
  local combo_catalog="$1"; shift
  local combo_n="$1"; shift
  local combos_per_gpu="$1"; shift
  local phase_prefix="$1"; shift
  local pass_label="$1"; shift
  local pass_desc="$1"; shift
  local seed_offset="$1"; shift

  local gpu_n="${#GPUS[@]}"
  local covered_combos
  local gpu_plan=()
  local gidx gpu planned_count

  covered_combos="$(compute_covered_combos "$combo_n" "$gpu_n" "$combos_per_gpu")"

  echo "=== [${ds}] RR-focused P2 ${pass_label} layout=L${layout_id} combos=${covered_combos}/${combo_n} gpus=${gpu_n} cpg=${combos_per_gpu} ==="
  if [ -n "$parent_result" ]; then
    echo "[${phase_prefix}] parent_result(L${layout_id})=${parent_result}"
  else
    echo "[${phase_prefix}] parent_result(L${layout_id})=(none found, start from script defaults)"
  fi
  for gidx in "${!GPUS[@]}"; do
    gpu="${GPUS[$gidx]}"
    planned_count="$(planned_combo_count_for_gpu "$gidx" "$covered_combos" "$gpu_n" "$combos_per_gpu")"
    gpu_plan+=("G${gpu}=${planned_count}")
  done
  echo "[${phase_prefix}] gpu_combo_plan=$(IFS=,; echo "${gpu_plan[*]}")"

  WORKER_PIDS=()
  for gidx in "${!GPUS[@]}"; do
    gpu="${GPUS[$gidx]}"
    (
      set -euo pipefail
      slot=0
      combo_idx=0
      for ((slot=0; slot<combos_per_gpu; slot++)); do
        combo_idx=$(( slot * gpu_n + gidx ))
        if [ "$combo_idx" -ge "$covered_combos" ]; then
          continue
        fi

        read -r emb dfeat dexp drouter expert_scale train_bs eval_bs <<< "$(read_combo "$combo_catalog" "$combo_idx")"
        read -r train_bs eval_bs <<< "$(apply_rr_oom_cap "$emb" "$dfeat" "$dexp" "$drouter" "$expert_scale" "$train_bs" "$eval_bs")"
        lr_space="$(lr_space_for_combo "$emb" "$dfeat" "$dexp" "$drouter")"
        wd_space="$(wd_space_for_combo "$emb" "$dfeat" "$dexp" "$drouter")"
        seed=$((SEED_BASE + seed_offset + layout_idx * 10000 + combo_idx))
        phase="${phase_prefix}_L${layout_id}_G${gpu}_C$(printf '%02d' "$combo_idx")_E${emb}_F${dfeat}_H${dexp}_R${drouter}_B${train_bs}"
        exp_name="${phase_prefix}_${ds}_${pass_label}_serial_layout${layout_id}"
        exp_desc="${pass_desc}"
        exp_focus="fmoe_v2_layout_id,fmoe_stage_execution_mode,embedding_size,d_feat_emb,d_expert_hidden,d_router_hidden,expert_scale,train_batch_size,eval_batch_size,hidden_dropout_prob,balance_loss_lambda,learning_rate,weight_decay"

        echo "[${phase_prefix}][${ds}] layout=L${layout_id} gpu=${gpu} combo=${combo_idx}/${combo_n} emb=${emb} dfeat=${dfeat} dexp=${dexp} drouter=${drouter} scale=${expert_scale} bs=${train_bs}/${eval_bs}"

        cmd=(
          bash "${SCRIPT_DIR}/tune_hparam.sh"
          --dataset "$ds"
          --layout-id "$layout_id"
          --execution serial
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
        if [ -n "$parent_result" ]; then
          cmd+=(--parent-result "$parent_result")
        fi
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
            retry_lr_space="$(lr_space_for_combo "$emb" "$dfeat" "$dexp" "$drouter")"
            retry_phase="${phase}_RBS${retry_train_bs}"
            retry_desc="${exp_desc} OOM-retry(train/eval=${retry_train_bs}/${retry_eval_bs})."
            echo "[${phase_prefix}][OOM-RETRY] ${ds} layout=L${layout_id} phase=${phase} -> ${retry_phase} bs ${train_bs}/${eval_bs} -> ${retry_train_bs}/${retry_eval_bs}"
            retry_cmd=(
              bash "${SCRIPT_DIR}/tune_hparam.sh"
              --dataset "$ds"
              --layout-id "$layout_id"
              --execution serial
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
              --lr-space "$retry_lr_space"
              --wd-space "$wd_space"
              --dropout-space "$DROP_SPACE"
              --balance-space "$BAL_SPACE"
              --exp-name "$exp_name"
              --exp-desc "$retry_desc"
              --exp-focus "$exp_focus"
            )
            if [ -n "$parent_result" ]; then
              retry_cmd+=(--parent-result "$parent_result")
            fi
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
    echo "[ERROR] RR-focused P2 failed for dataset=${ds}, layout=L${layout_id}, pass=${pass_label}" >&2
    return 1
  fi
  return 0
}

MAIN_EXP_DESC="${PHASE_PREFIX}: RR-focused serial P2 from strong P1 layouts; 80-combo sweep with anchor-near, high-capacity, and outlier dim/router/batch mixes."
SURPRISE_EXP_DESC="${SURPRISE_PHASE_PREFIX}: surprise-layout probe on high-upside non-anchor layouts from RR P1; compact anchor+outlier subset for fast contrast."

for ds in "${DATASET_ARR[@]}"; do
  surprise_label="${SURPRISE_LAYOUT_IDS:-none}"
  echo "=== [${ds}] RR-focused P2 start: main_layouts=${LAYOUT_IDS} main_combos=${MAIN_COMBO_N}/${MAIN_COVERED_COMBOS} surprise_layouts=${surprise_label} surprise_combos=${SURPRISE_COMBO_N}/${SURPRISE_COVERED_COMBOS} ==="

  for layout_idx in "${!LAYOUT_ARR[@]}"; do
    layout_id="${LAYOUT_ARR[$layout_idx]}"
    parent_result="$(find_best_parent_result "$ds" "$layout_id")"
    if ! run_layout_jobs_for_dataset \
      "$ds" \
      "$layout_idx" \
      "$layout_id" \
      "$parent_result" \
      "$COMBO_CATALOG" \
      "$MAIN_COMBO_N" \
      "$COMBOS_PER_GPU" \
      "$PHASE_PREFIX" \
      "main" \
      "$MAIN_EXP_DESC" \
      "0"; then
      exit 1
    fi
  done

  if [ "${#SURPRISE_LAYOUT_ARR[@]}" -gt 0 ] && [ "$SURPRISE_COMBO_N" -gt 0 ]; then
    for layout_idx in "${!SURPRISE_LAYOUT_ARR[@]}"; do
      layout_id="${SURPRISE_LAYOUT_ARR[$layout_idx]}"
      parent_result="$(find_best_parent_result "$ds" "$layout_id")"
      if ! run_layout_jobs_for_dataset \
        "$ds" \
        "$layout_idx" \
        "$layout_id" \
        "$parent_result" \
        "$SURPRISE_COMBO_CATALOG" \
        "$SURPRISE_COMBO_N" \
        "$SURPRISE_COMBOS_PER_GPU" \
        "$SURPRISE_PHASE_PREFIX" \
        "surprise" \
        "$SURPRISE_EXP_DESC" \
        "500000"; then
        exit 1
      fi
    done
  fi

  echo "=== [${ds}] RR-focused P2 done ==="
done

trap - INT TERM

run_update_track_report fmoe_v2
