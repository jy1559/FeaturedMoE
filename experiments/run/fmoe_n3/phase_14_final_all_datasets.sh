#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=/dev/null
source "${RUN_DIR}/common/run_metadata.sh"

DATASETS="KuaiRecLargeStrictPosV2_0.2,lastfm0.03,amazon_beauty,foursquare,movielens1m,retail_rocket"
GPU_LIST="0,1,2,3,4,5,6,7"
SEEDS="1,2,3,4"
SEED_BASE="91000"
ARCHITECTURES="A1,A2,A3,A4,A5"

COMMON_HPARAMS="H1,H3"
DEFAULT_OUTLIER_HPARAM="H4"
DATASET_OUTLIER_HPARAMS=""

MAX_EVALS="20"
TUNE_EPOCHS="100"
TUNE_PATIENCE="10"

FEATURE_GROUP_BIAS_LAMBDA="0.05"
RULE_BIAS_SCALE="0.1"
Z_LOSS_LAMBDA="1e-4"
BALANCE_LOSS_LAMBDA="0.0"
MACRO_HISTORY_WINDOW="5"
FAMILY_DROPOUT_PROB="0.10"
FEATURE_DROPOUT_PROB="0.0"
A2_ROUTE_CONSISTENCY_LAMBDA="8e-4"
A2_ROUTE_CONSISTENCY_MIN_SIM="0.995"
A2_Z_LOSS_LAMBDA="2e-4"
A4_INTRA_GROUP_BIAS_SCALE="0.12"

MAX_ITEM_LIST_LENGTH="20"
BATCH_SIZE="4096"
NUM_HEADS="4"
ATTN_DROPOUT_PROB="0.1"
D_FEAT_EMB="16"
EXPERT_SCALE="3"
SEARCH_LR_MIN="1.5e-4"
SEARCH_LR_MAX="8e-3"
SEARCH_LR_SCHEDULER="warmup_cosine"

MANIFEST_OUT=""
RESUME_FROM_LOGS="true"
VERIFY_LOGGING="true"

DRY_RUN="${DRY_RUN:-false}"
SMOKE_TEST="false"
SMOKE_MAX_RUNS="2"

usage() {
  cat <<USAGE
Usage: $0 [--datasets ...] [--gpus 0,1,2,3] [--seeds 1,2,3,4]
          [--architecture A1,A2,A3,A4,A5]
          [--common-hparams H1,H3] [--dataset-outlier-hparams "ds:H4,ds2:H6"]
          [--max-evals 20] [--tune-epochs 100] [--tune-patience 10]
          [--family-dropout-prob 0.10]
          [--manifest-out path] [--resume-from-logs|--no-resume-from-logs]
          [--verify-logging|--no-verify-logging]
          [--dry-run] [--smoke-test] [--smoke-max-runs 2]
USAGE
}

while [ $# -gt 0 ]; do
  case "$1" in
    --datasets) DATASETS="$2"; shift 2 ;;
    --gpus) GPU_LIST="$2"; shift 2 ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    --seed-base) SEED_BASE="$2"; shift 2 ;;
    --architecture|--architectures) ARCHITECTURES="$2"; shift 2 ;;
    --common-hparams) COMMON_HPARAMS="$2"; shift 2 ;;
    --default-outlier-hparam) DEFAULT_OUTLIER_HPARAM="$2"; shift 2 ;;
    --dataset-outlier-hparams) DATASET_OUTLIER_HPARAMS="$2"; shift 2 ;;
    --max-evals) MAX_EVALS="$2"; shift 2 ;;
    --tune-epochs) TUNE_EPOCHS="$2"; shift 2 ;;
    --tune-patience) TUNE_PATIENCE="$2"; shift 2 ;;
    --feature-group-bias-lambda) FEATURE_GROUP_BIAS_LAMBDA="$2"; shift 2 ;;
    --rule-bias-scale) RULE_BIAS_SCALE="$2"; shift 2 ;;
    --z-loss-lambda) Z_LOSS_LAMBDA="$2"; shift 2 ;;
    --balance-loss-lambda) BALANCE_LOSS_LAMBDA="$2"; shift 2 ;;
    --macro-history-window) MACRO_HISTORY_WINDOW="$2"; shift 2 ;;
    --family-dropout-prob) FAMILY_DROPOUT_PROB="$2"; shift 2 ;;
    --feature-dropout-prob) FEATURE_DROPOUT_PROB="$2"; shift 2 ;;
    --a2-route-consistency-lambda) A2_ROUTE_CONSISTENCY_LAMBDA="$2"; shift 2 ;;
    --a2-route-consistency-min-sim) A2_ROUTE_CONSISTENCY_MIN_SIM="$2"; shift 2 ;;
    --a2-z-loss-lambda) A2_Z_LOSS_LAMBDA="$2"; shift 2 ;;
    --a4-intra-group-bias-scale) A4_INTRA_GROUP_BIAS_SCALE="$2"; shift 2 ;;
    --max-item-list-length) MAX_ITEM_LIST_LENGTH="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --num-heads) NUM_HEADS="$2"; shift 2 ;;
    --attn-dropout-prob) ATTN_DROPOUT_PROB="$2"; shift 2 ;;
    --d-feat-emb) D_FEAT_EMB="$2"; shift 2 ;;
    --expert-scale) EXPERT_SCALE="$2"; shift 2 ;;
    --search-lr-min) SEARCH_LR_MIN="$2"; shift 2 ;;
    --search-lr-max) SEARCH_LR_MAX="$2"; shift 2 ;;
    --search-lr-scheduler) SEARCH_LR_SCHEDULER="$2"; shift 2 ;;
    --manifest-out) MANIFEST_OUT="$2"; shift 2 ;;
    --resume-from-logs) RESUME_FROM_LOGS="true"; shift ;;
    --no-resume-from-logs) RESUME_FROM_LOGS="false"; shift ;;
    --verify-logging) VERIFY_LOGGING="true"; shift ;;
    --no-verify-logging) VERIFY_LOGGING="false"; shift ;;
    --dry-run) DRY_RUN="true"; shift ;;
    --smoke-test) SMOKE_TEST="true"; shift ;;
    --smoke-max-runs) SMOKE_MAX_RUNS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
  esac
done

if [ -z "${RUN_PYTHON_BIN:-}" ] && [ -x "/venv/FMoE/bin/python" ]; then
  export RUN_PYTHON_BIN="/venv/FMoE/bin/python"
fi

PYTHON_BIN="${RUN_PYTHON_BIN:-$(run_python_bin)}"
CMD=(
  "${PYTHON_BIN}"
  "${SCRIPT_DIR}/run_final_all_datasets.py"
  --datasets "${DATASETS}"
  --gpus "${GPU_LIST}"
  --seeds "${SEEDS}"
  --seed-base "${SEED_BASE}"
  --architectures "${ARCHITECTURES}"
  --common-hparams "${COMMON_HPARAMS}"
  --default-outlier-hparam "${DEFAULT_OUTLIER_HPARAM}"
  --max-evals "${MAX_EVALS}"
  --tune-epochs "${TUNE_EPOCHS}"
  --tune-patience "${TUNE_PATIENCE}"
  --feature-group-bias-lambda "${FEATURE_GROUP_BIAS_LAMBDA}"
  --rule-bias-scale "${RULE_BIAS_SCALE}"
  --z-loss-lambda "${Z_LOSS_LAMBDA}"
  --balance-loss-lambda "${BALANCE_LOSS_LAMBDA}"
  --macro-history-window "${MACRO_HISTORY_WINDOW}"
  --family-dropout-prob "${FAMILY_DROPOUT_PROB}"
  --feature-dropout-prob "${FEATURE_DROPOUT_PROB}"
  --a2-route-consistency-lambda "${A2_ROUTE_CONSISTENCY_LAMBDA}"
  --a2-route-consistency-min-sim "${A2_ROUTE_CONSISTENCY_MIN_SIM}"
  --a2-z-loss-lambda "${A2_Z_LOSS_LAMBDA}"
  --a4-intra-group-bias-scale "${A4_INTRA_GROUP_BIAS_SCALE}"
  --max-item-list-length "${MAX_ITEM_LIST_LENGTH}"
  --batch-size "${BATCH_SIZE}"
  --num-heads "${NUM_HEADS}"
  --attn-dropout-prob "${ATTN_DROPOUT_PROB}"
  --d-feat-emb "${D_FEAT_EMB}"
  --expert-scale "${EXPERT_SCALE}"
  --search-lr-min "${SEARCH_LR_MIN}"
  --search-lr-max "${SEARCH_LR_MAX}"
  --search-lr-scheduler "${SEARCH_LR_SCHEDULER}"
  --smoke-max-runs "${SMOKE_MAX_RUNS}"
)

if [ -n "${DATASET_OUTLIER_HPARAMS}" ]; then
  CMD+=(--dataset-outlier-hparams "${DATASET_OUTLIER_HPARAMS}")
fi
if [ -n "${MANIFEST_OUT}" ]; then
  CMD+=(--manifest-out "${MANIFEST_OUT}")
fi
if [ "${RESUME_FROM_LOGS}" = "true" ]; then
  CMD+=(--resume-from-logs)
else
  CMD+=(--no-resume-from-logs)
fi
if [ "${VERIFY_LOGGING}" = "true" ]; then
  CMD+=(--verify-logging)
else
  CMD+=(--no-verify-logging)
fi
if [ "${DRY_RUN}" = "true" ]; then
  CMD+=(--dry-run)
fi
if [ "${SMOKE_TEST}" = "true" ]; then
  CMD+=(--smoke-test)
fi

run_echo_cmd "${CMD[@]}"
"${CMD[@]}"

echo "[All Done] Final_all_datasets runs completed: ${DATASETS}"
