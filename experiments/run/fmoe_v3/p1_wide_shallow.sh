#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=/dev/null
source "${RUN_DIR}/common/run_metadata.sh"
# shellcheck source=/dev/null
source "${RUN_DIR}/common/dispatch_gpu_queue.sh"

DATASETS="movielens1m"
GPU_LIST="0,1"
COMBOS_PER_GPU="3"
MIN_TOTAL_LAYERS="2"
MAX_TOTAL_LAYERS="6"
MAX_EVALS="12"
TUNE_EPOCHS="80"
TUNE_PATIENCE="8"
SEED_BASE="42"
PHASE_PREFIX="P1S"
SCHEDULE_PRESET="off"
SEARCH_PROFILE="p1_shallow"
SERIAL_LAYOUT_IDS="0,1,2,3,4,5,6,7,8,9,15,16,17,18,19"
PARALLEL_LAYOUT_IDS="10,11,12,13,14,20,21,22,23,24,25,26,27,28,29"
TRAIN_BATCH_SIZE="4096"
EVAL_BATCH_SIZE="4096"
LR_SPACE="1e-2,5e-3,2.5e-3,1e-3,5e-4,2.5e-4,1e-4"
WD_SPACE="0,1e-6,1e-5,1e-4,1e-3"
LOG_WANDB="false"
DRY_RUN="${DRY_RUN:-false}"
EXP_NAME_BASE="P1_wide_shallow"
EXP_DESC_BASE="Wide-shallow screen over serial/parallel layouts (fixed dims), tune LR/WD with small eval budget."
EXP_FOCUS="fmoe_stage_execution_mode,fmoe_v2_layout_id,learning_rate,weight_decay,train_batch_size,eval_batch_size,d_feat_emb,d_expert_hidden,d_router_hidden"

usage() {
  cat <<USAGE
Usage: $0 [--datasets movielens1m,retail_rocket] [--gpus 0,1]
          [--combos-per-gpu N] [--min-total-layers N] [--max-total-layers N] [--max-evals N]
          [--tune-epochs N] [--tune-patience N]
          [--serial-layout-ids csv] [--parallel-layout-ids csv]
          [--lr-space csv] [--wd-space csv] [--seed-base N] [--phase-prefix P1S]
          [--exp-name-base name] [--exp-desc-base text]
USAGE
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --datasets) DATASETS="$2"; shift 2 ;;
    --gpus) GPU_LIST="$2"; shift 2 ;;
    --combos-per-gpu) COMBOS_PER_GPU="$2"; shift 2 ;;
    --min-total-layers) MIN_TOTAL_LAYERS="$2"; shift 2 ;;
    --max-total-layers) MAX_TOTAL_LAYERS="$2"; shift 2 ;;
    --max-evals) MAX_EVALS="$2"; shift 2 ;;
    --tune-epochs) TUNE_EPOCHS="$2"; shift 2 ;;
    --tune-patience) TUNE_PATIENCE="$2"; shift 2 ;;
    --seed-base) SEED_BASE="$2"; shift 2 ;;
    --phase-prefix) PHASE_PREFIX="$2"; shift 2 ;;
    --schedule-preset) SCHEDULE_PRESET="$2"; shift 2 ;;
    --serial-layout-ids) SERIAL_LAYOUT_IDS="$2"; shift 2 ;;
    --parallel-layout-ids) PARALLEL_LAYOUT_IDS="$2"; shift 2 ;;
    --train-batch-size) TRAIN_BATCH_SIZE="$2"; shift 2 ;;
    --eval-batch-size) EVAL_BATCH_SIZE="$2"; shift 2 ;;
    --lr-space) LR_SPACE="$2"; shift 2 ;;
    --wd-space) WD_SPACE="$2"; shift 2 ;;
    --exp-name-base) EXP_NAME_BASE="$2"; shift 2 ;;
    --exp-desc-base) EXP_DESC_BASE="$2"; shift 2 ;;
    --log-wandb) LOG_WANDB="true"; shift ;;
    --no-wandb) LOG_WANDB="false"; shift ;;
    --dry-run) DRY_RUN="true"; shift ;;
    --help|-h) usage; exit 0 ;;
    --) shift; break ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

if ! [[ "$COMBOS_PER_GPU" =~ ^[0-9]+$ ]] || [ "$COMBOS_PER_GPU" -le 0 ]; then
  echo "--combos-per-gpu must be positive integer" >&2
  exit 1
fi
if ! [[ "$MAX_TOTAL_LAYERS" =~ ^[0-9]+$ ]] || [ "$MAX_TOTAL_LAYERS" -le 0 ]; then
  echo "--max-total-layers must be positive integer" >&2
  exit 1
fi
if ! [[ "$MIN_TOTAL_LAYERS" =~ ^[0-9]+$ ]] || [ "$MIN_TOTAL_LAYERS" -le 0 ]; then
  echo "--min-total-layers must be positive integer" >&2
  exit 1
fi
if [ "$MIN_TOTAL_LAYERS" -gt "$MAX_TOTAL_LAYERS" ]; then
  echo "--min-total-layers must be <= --max-total-layers" >&2
  exit 1
fi

if ! [[ "$MAX_EVALS" =~ ^[0-9]+$ ]] || [ "$MAX_EVALS" -le 0 ]; then
  echo "--max-evals must be positive integer" >&2
  exit 1
fi

dispatch_parse_csv "$GPU_LIST" GPUS
[ "${#GPUS[@]}" -eq 0 ] && { echo "Empty GPU list"; exit 1; }

dispatch_parse_csv "$DATASETS" DATASET_ARR
[ "${#DATASET_ARR[@]}" -eq 0 ] && { echo "Empty dataset list"; exit 1; }

EXP_DIR="$(run_experiments_dir)"
cd "$EXP_DIR"
run_export_runtime_env
PY_BIN="$(run_python_bin)"

INTERRUPTED="false"
WORKER_PIDS=()
on_interrupt() {
  INTERRUPTED="true"
  echo "[INTERRUPT] stopping p1 workers..."
  local p
  for p in "${WORKER_PIDS[@]:-}"; do
    kill -TERM "$p" 2>/dev/null || true
  done
  wait || true
  exit 130
}
trap on_interrupt INT TERM

LAYOUT_CONFIG_PATH="${EXP_DIR}/configs/model/featured_moe_v3_tune.yaml"

filter_layout_ids_for_mode() {
  local mode="$1"
  local raw_ids="$2"
  "$PY_BIN" - <<'PY' "$LAYOUT_CONFIG_PATH" "$mode" "$raw_ids" "$MIN_TOTAL_LAYERS" "$MAX_TOTAL_LAYERS"
import sys
import yaml

cfg_path = sys.argv[1]
mode = sys.argv[2].strip().lower()
raw_ids = sys.argv[3]
min_total_layers = int(sys.argv[4])
max_total_layers = int(sys.argv[5])

def parse_ids(raw):
    out = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(int(tok))
    return out

with open(cfg_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}

catalog = (
    ((cfg.get("layout_execution") or {}).get("fmoe_v3_layout_catalog"))
    or cfg.get("fmoe_v3_layout_catalog")
    or []
)
if not isinstance(catalog, list):
    raise SystemExit("invalid fmoe_v3_layout_catalog")

for lid in parse_ids(raw_ids):
    if lid < 0 or lid >= len(catalog):
        continue
    entry = catalog[lid] or {}
    if str(entry.get("execution", "serial")).strip().lower() != mode:
        continue

    pre = int(entry.get("global_pre_layers", 0) or 0)
    post = int(entry.get("global_post_layers", 0) or 0)
    total = pre + post
    moe_total = 0
    stages = entry.get("stages") or {}
    for s in ("macro", "mid", "micro"):
        spec = stages.get(s) or {}
        p = int(spec.get("pass_layers", 0) or 0)
        m = int(spec.get("moe_blocks", 0) or 0)
        total += p + m
        moe_total += m

    if min_total_layers <= total <= max_total_layers:
        print(f"{lid}:{total}:{moe_total}")
PY
}

parse_filtered_ids() {
  local mode="$1"
  local raw_ids="$2"
  local -n out_ids_ref="$3"
  local -n out_desc_ref="$4"

  local rows
  mapfile -t rows < <(filter_layout_ids_for_mode "$mode" "$raw_ids")
  out_ids_ref=()
  out_desc_ref=()
  local row lid total moe
  for row in "${rows[@]}"; do
    IFS=':' read -r lid total moe <<< "$row"
    [ -z "${lid:-}" ] && continue
    out_ids_ref+=("$lid")
    out_desc_ref+=("L${lid}(layers=${total},moe=${moe})")
  done
}

SERIAL_IDS=()
SERIAL_DESC=()
PARALLEL_IDS=()
PARALLEL_DESC=()

parse_filtered_ids "serial" "$SERIAL_LAYOUT_IDS" SERIAL_IDS SERIAL_DESC
parse_filtered_ids "parallel" "$PARALLEL_LAYOUT_IDS" PARALLEL_IDS PARALLEL_DESC

MODE_ORDER=()
if [ "${#SERIAL_IDS[@]}" -gt 0 ]; then
  MODE_ORDER+=("serial")
fi
if [ "${#PARALLEL_IDS[@]}" -gt 0 ]; then
  MODE_ORDER+=("parallel")
fi

if [ "${#MODE_ORDER[@]}" -eq 0 ]; then
  echo "No available layouts after filtering (total_layers=${MIN_TOTAL_LAYERS}..${MAX_TOTAL_LAYERS})." >&2
  exit 1
fi

if [ "${#SERIAL_IDS[@]}" -gt 0 ]; then
  echo "[P1] serial layouts (${MIN_TOTAL_LAYERS}..${MAX_TOTAL_LAYERS}): ${SERIAL_DESC[*]}"
else
  echo "[P1] serial layouts (${MIN_TOTAL_LAYERS}..${MAX_TOTAL_LAYERS}): (none)"
fi
if [ "${#PARALLEL_IDS[@]}" -gt 0 ]; then
  echo "[P1] parallel layouts (${MIN_TOTAL_LAYERS}..${MAX_TOTAL_LAYERS}): ${PARALLEL_DESC[*]}"
else
  echo "[P1] parallel layouts (${MIN_TOTAL_LAYERS}..${MAX_TOTAL_LAYERS}): (none)"
fi

for ds in "${DATASET_ARR[@]}"; do
  modes_count="${#MODE_ORDER[@]}"
  total_jobs=$(( ${#GPUS[@]} * COMBOS_PER_GPU * modes_count ))

  echo "=== [${ds}] P1 wide-shallow (${total_jobs} runs = ${#GPUS[@]} gpus x ${COMBOS_PER_GPU} per mode x ${modes_count} modes) ==="

  WORKER_PIDS=()
  for gidx in "${!GPUS[@]}"; do
    gpu="${GPUS[$gidx]}"
    (
      set -euo pipefail
      mode_count="${#MODE_ORDER[@]}"
      for mode_idx in "${!MODE_ORDER[@]}"; do
        execution="${MODE_ORDER[$mode_idx]}"
        if [ "$execution" = "serial" ]; then
          MODE_IDS=("${SERIAL_IDS[@]}")
        else
          MODE_IDS=("${PARALLEL_IDS[@]}")
        fi
        [ "${#MODE_IDS[@]}" -eq 0 ] && continue

        for slot in $(seq 0 $((COMBOS_PER_GPU - 1))); do
          slot_global=$(( gidx * COMBOS_PER_GPU + slot ))
          lid_idx=$(( slot_global % ${#MODE_IDS[@]} ))
          layout_id="${MODE_IDS[$lid_idx]}"

          idx=$(( gidx * mode_count * COMBOS_PER_GPU + mode_idx * COMBOS_PER_GPU + slot ))
          seed=$(( SEED_BASE + idx ))
          phase="${PHASE_PREFIX}_G${gpu}_C$((slot+1))_${execution}_L${layout_id}"

          echo "[P1][${ds}] gpu=${gpu} mode=${execution} slot=$((slot+1))/${COMBOS_PER_GPU} layout=${layout_id} seed=${seed}"

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
            --search-profile "$SEARCH_PROFILE"
            --train-batch-size "$TRAIN_BATCH_SIZE"
            --eval-batch-size "$EVAL_BATCH_SIZE"
            --lr-space "$LR_SPACE"
            --wd-space "$WD_SPACE"
            --exp-name "${EXP_NAME_BASE}_${ds}"
            --exp-desc "${EXP_DESC_BASE}"
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

          "${cmd[@]}"
        done
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
    echo "[ERROR] P1 wide-shallow failed for dataset=${ds}" >&2
    exit 1
  fi

  echo "=== [${ds}] P1 wide-shallow done ==="
done

trap - INT TERM

if [ "$INTERRUPTED" = "true" ]; then
  exit 130
fi

run_update_track_report fmoe_v3
