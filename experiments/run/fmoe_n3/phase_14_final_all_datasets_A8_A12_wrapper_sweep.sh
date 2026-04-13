#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=/dev/null
source "${RUN_DIR}/common/run_metadata.sh"

DATASETS="KuaiRecLargeStrictPosV2_0.2,lastfm0.03,amazon_beauty,foursquare,movielens1m,retail_rocket"
GPU_LIST="0,1,2,3,4,5,6,7"
SEEDS="1"
SEED_BASE="97000"
ARCHITECTURES="A8,A10,A11,A12"

COMMON_HPARAMS="AUTO4"
DEFAULT_OUTLIER_HPARAM="H4"
DATASET_OUTLIER_HPARAMS=""

MAX_EVALS="12"
TUNE_EPOCHS="100"
TUNE_PATIENCE="10"

FEATURE_GROUP_BIAS_LAMBDA="0.05"
RULE_BIAS_SCALE="0.10"
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
ATTN_DROPOUT_PROB="0.10"
D_FEAT_EMB="16"
EXPERT_SCALE="3"
SEARCH_LR_SCHEDULER="warmup_cosine"

MANIFEST_OUT=""
RESUME_FROM_LOGS="true"
VERIFY_LOGGING="true"
DRY_RUN="${DRY_RUN:-false}"
SMOKE_TEST="false"
SMOKE_MAX_RUNS="4"

usage() {
  cat <<USAGE
Usage: $0 [--datasets ...] [--gpus 0,1,2,3] [--seeds 1,2]
          [--max-evals 12] [--tune-epochs 100] [--tune-patience 10]
          [--family-dropout-prob 0.10] [--attn-dropout-prob 0.10]
          [--dry-run] [--smoke-test] [--smoke-max-runs 4]
USAGE
}

while [ $# -gt 0 ]; do
  case "$1" in
    --datasets) DATASETS="$2"; shift 2 ;;
    --gpus) GPU_LIST="$2"; shift 2 ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    --seed-base) SEED_BASE="$2"; shift 2 ;;
    --max-evals) MAX_EVALS="$2"; shift 2 ;;
    --tune-epochs) TUNE_EPOCHS="$2"; shift 2 ;;
    --tune-patience) TUNE_PATIENCE="$2"; shift 2 ;;
    --family-dropout-prob) FAMILY_DROPOUT_PROB="$2"; shift 2 ;;
    --attn-dropout-prob) ATTN_DROPOUT_PROB="$2"; shift 2 ;;
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

IFS=',' read -r -a DATASET_ARRAY <<< "${DATASETS}"

csv_count() {
  local raw="$1"
  local -a items=()
  IFS=',' read -r -a items <<< "${raw}"
  echo "${#items[@]}"
}

dataset_hparam_count() {
  case "$1" in
    retail_rocket) echo "4" ;;
    KuaiRecLargeStrictPosV2_0.2|lastfm0.03|amazon_beauty|foursquare|movielens1m) echo "4" ;;
    *) echo "4" ;;
  esac
}

dataset_total_runs() {
  local dataset="$1"
  local arch_count seed_count total
  arch_count="$(csv_count "${ARCHITECTURES}")"
  seed_count="$(csv_count "${SEEDS}")"
  total=$(( $(dataset_hparam_count "${dataset}") * arch_count * seed_count ))
  if [ "${SMOKE_TEST}" = "true" ] && [ "${SMOKE_MAX_RUNS}" -lt "${total}" ]; then
    total="${SMOKE_MAX_RUNS}"
  fi
  echo "${total}"
}

dataset_lr_min() {
  case "$1" in
    KuaiRecLargeStrictPosV2_0.2) echo "2.5e-4" ;;
    lastfm0.03) echo "2.0e-4" ;;
    amazon_beauty) echo "4.5e-4" ;;
    foursquare) echo "1.5e-3" ;;
    movielens1m) echo "9.0e-4" ;;
    retail_rocket) echo "7.0e-4" ;;
    *) echo "3.0e-4" ;;
  esac
}

dataset_lr_max() {
  case "$1" in
    KuaiRecLargeStrictPosV2_0.2) echo "1.2e-3" ;;
    lastfm0.03) echo "1.2e-3" ;;
    amazon_beauty) echo "2.2e-3" ;;
    foursquare) echo "6.0e-3" ;;
    movielens1m) echo "6.0e-3" ;;
    retail_rocket) echo "3.0e-3" ;;
    *) echo "2.0e-3" ;;
  esac
}

TOTAL_RUNS="0"
for DATASET in "${DATASET_ARRAY[@]}"; do
  TOTAL_RUNS=$(( TOTAL_RUNS + $(dataset_total_runs "${DATASET}") ))
done

COMPLETED_BASE="0"
DATASET_COUNT="${#DATASET_ARRAY[@]}"

for DATASET_INDEX in "${!DATASET_ARRAY[@]}"; do
  DATASET="${DATASET_ARRAY[${DATASET_INDEX}]}"
  SEARCH_LR_MIN="$(dataset_lr_min "${DATASET}")"
  SEARCH_LR_MAX="$(dataset_lr_max "${DATASET}")"
  DATASET_RUNS="$(dataset_total_runs "${DATASET}")"

  export SLACK_NOTIFY_TOTAL_RUNS="${TOTAL_RUNS}"
  export SLACK_NOTIFY_GLOBAL_DONE_BASE="${COMPLETED_BASE}"
  export SLACK_NOTIFY_SCOPE_LABEL="dataset $((DATASET_INDEX + 1))/${DATASET_COUNT} ${DATASET}"

  CMD=(
    "${PYTHON_BIN}"
    "${SCRIPT_DIR}/run_final_a8_a12_wrapper_sweep.py"
    --datasets "${DATASET}"
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
    --no-migrate-existing-layout
  )

  if [ -n "${DATASET_OUTLIER_HPARAMS}" ]; then
    CMD+=(--dataset-outlier-hparams "${DATASET_OUTLIER_HPARAMS}")
  fi
  if [ -n "${MANIFEST_OUT}" ]; then
    CMD+=(--manifest-out "${MANIFEST_OUT}_${DATASET}")
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

echo "[A8/A10-A12 wrapper sweep] dataset=${DATASET} lr=[${SEARCH_LR_MIN},${SEARCH_LR_MAX}] hparams=${COMMON_HPARAMS}"
  run_echo_cmd "${CMD[@]}"
  "${CMD[@]}"
  COMPLETED_BASE=$(( COMPLETED_BASE + DATASET_RUNS ))
done

echo "[All Done] Final_all_datasets A8/A10~A12 wrapper sweep completed: ${DATASETS}"
