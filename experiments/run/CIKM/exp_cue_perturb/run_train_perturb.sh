#!/usr/bin/env bash
# Train-time cue perturbation ablation (구조 변경 → 새로 학습)
# Usage:
#   bash run_train_perturb.sh 0              # GPU 0, 모든 데이터셋/조건
#   bash run_train_perturb.sh 0 1            # GPU 0: KuaiRec, GPU 1: foursquare (round-robin)
#   bash run_train_perturb.sh 0 --datasets KuaiRec
#   bash run_train_perturb.sh 0 --conditions hidden_only feature_only

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# GPU 인수 파싱 (-- 옵션 전까지)
GPUS=()
EXTRA=()
PARSING_GPUS=true

for arg in "$@"; do
    if [[ "$arg" == --* ]]; then
        PARSING_GPUS=false
    fi
    if $PARSING_GPUS; then
        GPUS+=("$arg")
    else
        EXTRA+=("$arg")
    fi
done

if [ ${#GPUS[@]} -eq 0 ]; then
    GPUS=("0")
fi

CMD=(python train_perturb.py --gpus "${GPUS[@]}" "${EXTRA[@]}")

echo "[run_train_perturb] GPUs=${GPUS[*]}"
echo "[run_train_perturb] cmd: ${CMD[*]}"
echo ""

"${CMD[@]}"
