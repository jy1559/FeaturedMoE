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

usage() {
  cat <<USAGE
Usage: $0 [--datasets movielens1m,retail_rocket] [--gpus 0,1] [--seed-base 42]
USAGE
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --datasets) DATASETS="$2"; shift 2 ;;
    --gpus) GPU_LIST="$2"; shift 2 ;;
    --seed-base) SEED_BASE="$2"; shift 2 ;;
    --skip-p4) RUN_P4="false"; shift ;;
    --dry-run) DRY_RUN="true"; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

EXP_DIR="$(run_experiments_dir)"
cd "$EXP_DIR"
run_export_runtime_env

IFS=',' read -r -a DS_ARR <<< "$DATASETS"
IFS=',' read -r -a GPU_ARR <<< "$GPU_LIST"
[ "${#GPU_ARR[@]}" -eq 0 ] && { echo "Empty GPU list"; exit 1; }

DRY_ARGS=()
if [ "$DRY_RUN" = "true" ]; then
  DRY_ARGS+=(--dry-run)
fi

layout_default() {
  case "$1" in
    movielens1m) echo 7 ;;
    retail_rocket) echo 3 ;;
    *) echo 0 ;;
  esac
}

p1_trials() {
  case "$1" in
    movielens1m) echo 40 ;;
    retail_rocket) echo 30 ;;
    *) echo 20 ;;
  esac
}

latest_fmoe_result() {
  local ds="$1"
  local axis="$2"
  python - <<'PY' "$ds" "$axis"
from pathlib import Path
import json,os,sys

ds=sys.argv[1]
axis=sys.argv[2]
base=Path(os.environ.get('HYPEROPT_RESULTS_DIR', 'run/artifacts/results'))
root=base / 'fmoe'
if not root.exists():
    print('')
    raise SystemExit
cands=[]
for p in root.glob(f"{ds}_FeaturedMoE_*.json"):
    try:
        d=json.load(open(p,'r',encoding='utf-8'))
    except Exception:
        continue
    if axis and str(d.get('run_axis','')).lower()!=axis.lower():
        continue
    cands.append(p)
if not cands:
    print('')
    raise SystemExit
cands.sort(key=lambda x:x.stat().st_mtime, reverse=True)
print(str(cands[0]))
PY
}

layout_from_result() {
  local p="$1"
  python - <<'PY' "$p"
import json,sys
p=sys.argv[1]
d=json.load(open(p,'r',encoding='utf-8'))
bp=d.get('best_params') or {}
layout=bp.get('arch_layout_id')
if layout is None:
    layout=(d.get('fixed_search') or {}).get('arch_layout_id',0)
print(layout)
PY
}

for i in "${!DS_ARR[@]}"; do
  ds="${DS_ARR[$i]}"
  gpu="${GPU_ARR[$((i % ${#GPU_ARR[@]}))]}"
  l0="$(layout_default "$ds")"

  echo "=== [${ds}] P0: schedule-off reproducibility (3 seeds) ==="
  for s in 0 1 2; do
    seed=$((SEED_BASE + s))
    bash "${SCRIPT_DIR}/train_single.sh" \
      --dataset "$ds" --gpu "$gpu" --seed "$seed" --layout-id "$l0" --schedule off --phase P0 "${DRY_ARGS[@]}"
  done

  echo "=== [${ds}] P1: hparam tuning ==="
  bash "${SCRIPT_DIR}/tune_hparam.sh" \
    --dataset "$ds" --layout_id "$l0" --schedule off --gpu "$gpu" \
    --max-evals "$(p1_trials "$ds")" --phase P1 --seed "$SEED_BASE" "${DRY_ARGS[@]}"
  if [ "$DRY_RUN" = "true" ]; then
    continue
  fi
  p1_result="$(latest_fmoe_result "$ds" hparam)"
  [ -z "$p1_result" ] && { echo "No P1 result for $ds"; exit 1; }

  echo "=== [${ds}] P2: layout tuning ==="
  bash "${SCRIPT_DIR}/tune_layout.sh" \
    --dataset "$ds" --parent_result "$p1_result" --gpu "$gpu" --max-evals 20 --phase P2 --seed "$SEED_BASE" "${DRY_ARGS[@]}"
  p2_result="$(latest_fmoe_result "$ds" layout)"
  [ -z "$p2_result" ] && { echo "No P2 result for $ds"; exit 1; }

  l3="$(layout_from_result "$p2_result")"

  echo "=== [${ds}] P3: hparam refinement with best layout=${l3} ==="
  bash "${SCRIPT_DIR}/tune_hparam.sh" \
    --dataset "$ds" --layout_id "$l3" --schedule off --gpu "$gpu" \
    --max-evals 20 --phase P3 --seed "$SEED_BASE" --parent-result "$p2_result" "${DRY_ARGS[@]}"
  p3_result="$(latest_fmoe_result "$ds" hparam)"
  [ -z "$p3_result" ] && { echo "No P3 result for $ds"; exit 1; }

  if [ "$RUN_P4" = "true" ]; then
    echo "=== [${ds}] P4: schedule axis split tuning ==="
    bash "${SCRIPT_DIR}/tune_schedule.sh" --dataset "$ds" --parent_result "$p3_result" --mode alpha --gpu "$gpu" --phase P4 --seed "$SEED_BASE" "${DRY_ARGS[@]}"
    bash "${SCRIPT_DIR}/tune_schedule.sh" --dataset "$ds" --parent_result "$p3_result" --mode temp --gpu "$gpu" --phase P4 --seed "$SEED_BASE" "${DRY_ARGS[@]}"
    bash "${SCRIPT_DIR}/tune_schedule.sh" --dataset "$ds" --parent_result "$p3_result" --mode topk --gpu "$gpu" --phase P4 --seed "$SEED_BASE" "${DRY_ARGS[@]}"
    bash "${SCRIPT_DIR}/tune_schedule.sh" --dataset "$ds" --parent_result "$p3_result" --mode combined --gpu "$gpu" --phase P4 --seed "$SEED_BASE" "${DRY_ARGS[@]}"
  fi

  echo "=== [${ds}] done ==="
done
