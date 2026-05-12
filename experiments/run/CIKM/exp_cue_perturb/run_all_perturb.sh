#!/usr/bin/env bash
# 전체 cue perturbation 실험 실행 (A12 architecture)
#
# GPU global queue 방식:
#   - P0:     catalog top-N hparam × 데이터셋 → GPU global queue 병렬 학습
#   - Eval:   P0 best-K checkpoint × eval conditions → GPU global queue 병렬
#   - Train:  조건 × 데이터셋 × hparam → GPU global queue 병렬
#
# Usage:
#   bash run_all_perturb.sh 0 1 2 3                              # GPU 4개, 모든 데이터셋
#   bash run_all_perturb.sh 0 1 2 3 --datasets foursquare        # foursquare만
#   bash run_all_perturb.sh 0 1 2 3 --datasets foursquare KuaiRec
#   bash run_all_perturb.sh 0 1 2 3 --p0-candidates 4 --top-k 2
#   bash run_all_perturb.sh 0 1 2 3 --skip-p0                    # P0 건너뛰기
#   bash run_all_perturb.sh 0 1 2 3 --skip-eval                  # eval perturb 건너뛰기
#   bash run_all_perturb.sh 0 1 2 3 --skip-train                 # train perturb 건너뛰기
#
# 기본값:
#   --p0-candidates 4   카탈로그에서 hparam 후보 4개
#   --top-k 2           P0 결과에서 best 2개 checkpoint 사용

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── 인수 파싱 ──────────────────────────────────────────────────────────────────
GPUS=()
DATASETS_ARG=""
P0_CANDIDATES="4"
TOP_K="2"
SKIP_P0=""
SKIP_EVAL=""
SKIP_TRAIN=""

i=0
args=("$@")
while [ $i -lt ${#args[@]} ]; do
    arg="${args[$i]}"
    case "$arg" in
        --datasets)
            i=$((i + 1))
            DS_LIST=()
            while [ $i -lt ${#args[@]} ] && [[ "${args[$i]}" != --* ]] && ! [[ "${args[$i]}" =~ ^[0-9]+$ ]]; do
                DS_LIST+=("${args[$i]}")
                i=$((i + 1))
            done
            DATASETS_ARG="--datasets ${DS_LIST[*]}"
            continue
            ;;
        --p0-candidates)
            i=$((i + 1)); P0_CANDIDATES="${args[$i]}";;
        --top-k)
            i=$((i + 1)); TOP_K="${args[$i]}";;
        --skip-p0)    SKIP_P0="--skip-p0";;
        --skip-eval)  SKIP_EVAL="1";;
        --skip-train) SKIP_TRAIN="1";;
        [0-9]*)       GPUS+=("$arg");;
    esac
    i=$((i + 1))
done

if [ ${#GPUS[@]} -eq 0 ]; then
    GPUS=("0")
fi

GPUS_STR="${GPUS[*]}"

echo "══════════════════════════════════════════════════════════"
echo "[run_all_perturb] GPUs: ${GPUS_STR}"
echo "  datasets  : ${DATASETS_ARG:-all}"
echo "  p0-candidates: ${P0_CANDIDATES}  top-k: ${TOP_K}"
echo "  skip: p0=${SKIP_P0:+yes}  eval=${SKIP_EVAL:+yes}  train=${SKIP_TRAIN:+yes}"
echo "══════════════════════════════════════════════════════════"
echo ""

# ── Step 1: eval_perturb (P0 + eval conditions) ───────────────────────────────
if [ -z "$SKIP_EVAL" ]; then
    echo "── Step 1/3: eval_perturb (P0 → eval conditions) ──────"
    python eval_perturb.py \
        --gpus ${GPUS_STR} \
        ${DATASETS_ARG} \
        --p0-candidates "${P0_CANDIDATES}" \
        --top-k "${TOP_K}" \
        ${SKIP_P0}
    echo ""
else
    echo "── Step 1/3: eval_perturb 건너뜀 (--skip-eval)"
    echo ""
fi

# ── Step 2: train_perturb ─────────────────────────────────────────────────────
if [ -z "$SKIP_TRAIN" ]; then
    echo "── Step 2/3: train_perturb (조건별 학습) ───────────────"
    python train_perturb.py \
        --gpus ${GPUS_STR} \
        ${DATASETS_ARG} \
        --top-k "${TOP_K}"
    echo ""
else
    echo "── Step 2/3: train_perturb 건너뜀 (--skip-train)"
    echo ""
fi

# ── Step 3: 결과 취합 ─────────────────────────────────────────────────────────
echo "── Step 3/3: collect_results ────────────────────────────"
python collect_results.py

echo ""
echo "══════════════════════════════════════════════════════════"
echo "[run_all_perturb] Done."
echo "══════════════════════════════════════════════════════════"
