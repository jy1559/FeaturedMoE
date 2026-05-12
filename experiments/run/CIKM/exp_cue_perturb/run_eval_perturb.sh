#!/usr/bin/env bash
# Eval-only cue perturbation ablation (P0 checkpoint 재사용, 새 학습 없음)
# Usage:
#   bash run_eval_perturb.sh 0               # GPU 0, 모든 데이터셋
#   bash run_eval_perturb.sh 0 KuaiRec       # 특정 데이터셋만
#   bash run_eval_perturb.sh 0 KuaiRec foursquare eval_zero eval_shuffle

set -euo pipefail

GPU="${1:-0}"
shift || true

DATASETS=()
CONDITIONS=()
PARSING="datasets"

for arg in "$@"; do
    case "$arg" in
        eval_*)
            PARSING="conditions"
            CONDITIONS+=("$arg")
            ;;
        *)
            if [ "$PARSING" = "datasets" ]; then
                DATASETS+=("$arg")
            else
                CONDITIONS+=("$arg")
            fi
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CMD=(python eval_perturb.py --gpu "$GPU")

if [ ${#DATASETS[@]} -gt 0 ]; then
    CMD+=(--datasets "${DATASETS[@]}")
fi

if [ ${#CONDITIONS[@]} -gt 0 ]; then
    CMD+=(--conditions "${CONDITIONS[@]}")
fi

echo "[run_eval_perturb] GPU=$GPU"
echo "[run_eval_perturb] cmd: ${CMD[*]}"
echo ""

"${CMD[@]}"
