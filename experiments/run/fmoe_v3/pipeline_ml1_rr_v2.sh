#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=/dev/null
source "${RUN_DIR}/common/run_metadata.sh"

DATASETS="movielens1m,retail_rocket"
GPU_LIST="0,1"
SEED_BASE="42"
RUN_P4="true"
DRY_RUN="${DRY_RUN:-false}"
P1_COMBOS_PER_GPU="3"

usage() {
  cat <<USAGE
Usage: $0 [--datasets movielens1m,retail_rocket] [--gpus 0,1] [--seed-base 42]
          [--p1-combos-per-gpu 3]
USAGE
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --datasets) DATASETS="$2"; shift 2 ;;
    --gpus) GPU_LIST="$2"; shift 2 ;;
    --seed-base) SEED_BASE="$2"; shift 2 ;;
    --p1-combos-per-gpu) P1_COMBOS_PER_GPU="$2"; shift 2 ;;
    --skip-p4) RUN_P4="false"; shift ;;
    --dry-run) DRY_RUN="true"; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

EXP_DIR="$(run_experiments_dir)"
cd "$EXP_DIR"
run_export_runtime_env
PY_BIN="$(run_python_bin)"

IFS=',' read -r -a DS_ARR <<< "$DATASETS"
IFS=',' read -r -a GPU_ARR <<< "$GPU_LIST"
[ "${#GPU_ARR[@]}" -eq 0 ] && { echo "Empty GPU list"; exit 1; }

DRY_ARGS=()
if [ "$DRY_RUN" = "true" ]; then
  DRY_ARGS+=(--dry-run)
fi

layout_default() {
  case "$1" in
    movielens1m) echo 0 ;;
    retail_rocket) echo 0 ;;
    *) echo 0 ;;
  esac
}

execution_default() {
  case "$1" in
    movielens1m) echo serial ;;
    retail_rocket) echo serial ;;
    *) echo serial ;;
  esac
}

p1_trials() {
  case "$1" in
    movielens1m) echo 12 ;;
    retail_rocket) echo 8 ;;
    *) echo 8 ;;
  esac
}

best_fmoe_v3_result() {
  local ds="$1"
  local axis="$2"
  local phase_prefix="${3:-}"
  "$PY_BIN" - <<'PY' "$ds" "$axis" "$phase_prefix"
from pathlib import Path
import json, os, sys

ds = sys.argv[1]
axis = sys.argv[2]
phase_prefix = sys.argv[3]
base = Path(os.environ.get('HYPEROPT_RESULTS_DIR', 'run/artifacts/results'))
root = base / 'fmoe_v3'
if not root.exists():
    print('')
    raise SystemExit

cands = []
for p in root.glob(f"{ds}_FeaturedMoE_v3*.json"):
    try:
        d = json.load(open(p, 'r', encoding='utf-8'))
    except Exception:
        continue
    if axis and str(d.get('run_axis', '')).lower() != axis.lower():
        continue
    if phase_prefix and not str(d.get('run_phase', '')).startswith(phase_prefix):
        continue

    score = d.get('best_mrr@20')
    if not isinstance(score, (int, float)):
        bvr = d.get('best_valid_result', {})
        if isinstance(bvr, dict) and isinstance(bvr.get('mrr@20'), (int, float)):
            score = float(bvr['mrr@20'])
        else:
            score = float('-inf')
    cands.append((float(score), p.stat().st_mtime, p))

if not cands:
    print('')
    raise SystemExit

cands.sort(key=lambda x: (x[0], x[1]), reverse=True)
print(str(cands[0][2]))
PY
}

best_layout_exec_from_result() {
  local p="$1"
  "$PY_BIN" - <<'PY' "$p"
import json, sys
p = sys.argv[1]
d = json.load(open(p, 'r', encoding='utf-8'))
bp = d.get('best_params') or {}
fs = d.get('fixed_search') or {}
layout = bp.get('fmoe_v2_layout_id', fs.get('fmoe_v2_layout_id', 0))
exec_mode = bp.get('fmoe_stage_execution_mode', fs.get('fmoe_stage_execution_mode', 'serial'))
print(layout, exec_mode)
PY
}

for i in "${!DS_ARR[@]}"; do
  ds="${DS_ARR[$i]}"
  gpu="${GPU_ARR[$((i % ${#GPU_ARR[@]}))]}"
  l0="$(layout_default "$ds")"
  e0="$(execution_default "$ds")"

  echo "=== [${ds}] P0: schedule-off reproducibility (2 seeds) ==="
  for s in 0 1; do
    seed=$((SEED_BASE + s))
    bash "${SCRIPT_DIR}/train_single.sh" \
      --dataset "$ds" --gpu "$gpu" --seed "$seed" --layout-id "$l0" --execution "$e0" \
      --schedule off --phase P0 "${DRY_ARGS[@]}"
  done

  echo "=== [${ds}] P1: wide-shallow layout/execution screening ==="
  bash "${SCRIPT_DIR}/p1_wide_shallow.sh" \
    --datasets "$ds" \
    --gpus "$GPU_LIST" \
    --combos-per-gpu "$P1_COMBOS_PER_GPU" \
    --max-evals "$(p1_trials "$ds")" \
    --seed-base "$SEED_BASE" \
    --phase-prefix P1 \
    "${DRY_ARGS[@]}"

  if [ "$DRY_RUN" = "true" ]; then
    continue
  fi

  p1_result="$(best_fmoe_v3_result "$ds" hparam P1)"
  [ -z "$p1_result" ] && { echo "No P1 result for $ds"; exit 1; }

  echo "=== [${ds}] P2: layout/execution tuning ==="
  bash "${SCRIPT_DIR}/tune_layout.sh" \
    --dataset "$ds" --parent-result "$p1_result" --gpu "$gpu" --max-evals 20 --phase P2 --seed "$SEED_BASE" "${DRY_ARGS[@]}"

  p2_result="$(best_fmoe_v3_result "$ds" layout P2)"
  [ -z "$p2_result" ] && { echo "No P2 result for $ds"; exit 1; }

  read -r l3 e3 <<< "$(best_layout_exec_from_result "$p2_result")"

  echo "=== [${ds}] P3: hparam refinement with best layout=${l3} exec=${e3} ==="
  bash "${SCRIPT_DIR}/tune_hparam.sh" \
    --dataset "$ds" --layout-id "$l3" --execution "$e3" --schedule-preset off --gpu "$gpu" \
    --max-evals 20 --phase P3 --seed "$SEED_BASE" --parent-result "$p2_result" "${DRY_ARGS[@]}"

  p3_result="$(best_fmoe_v3_result "$ds" hparam P3)"
  [ -z "$p3_result" ] && { echo "No P3 result for $ds"; exit 1; }

  if [ "$RUN_P4" = "true" ]; then
    echo "=== [${ds}] P4: schedule axis split tuning ==="
    bash "${SCRIPT_DIR}/tune_schedule.sh" --dataset "$ds" --parent-result "$p3_result" --mode alpha --gpu "$gpu" --phase P4 --seed "$SEED_BASE" "${DRY_ARGS[@]}"
    bash "${SCRIPT_DIR}/tune_schedule.sh" --dataset "$ds" --parent-result "$p3_result" --mode temp --gpu "$gpu" --phase P4 --seed "$SEED_BASE" "${DRY_ARGS[@]}"
    bash "${SCRIPT_DIR}/tune_schedule.sh" --dataset "$ds" --parent-result "$p3_result" --mode topk --gpu "$gpu" --phase P4 --seed "$SEED_BASE" "${DRY_ARGS[@]}"
    bash "${SCRIPT_DIR}/tune_schedule.sh" --dataset "$ds" --parent-result "$p3_result" --mode combined --gpu "$gpu" --phase P4 --seed "$SEED_BASE" "${DRY_ARGS[@]}"
  fi

  echo "=== [${ds}] done ==="
done

if [ "$DRY_RUN" != "true" ]; then
  run_update_track_report fmoe_v3
fi
