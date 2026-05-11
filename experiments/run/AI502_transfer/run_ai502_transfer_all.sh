#!/usr/bin/env bash
set -euo pipefail

# 전체 AI502 transfer pipeline을 3개 그룹 barrier와 함께 실행한다.
# 시작 시 기존 artifacts 산출물을 비워서 이전 버그 결과가 섞이지 않게 한다.
# 기본값은 A12 기준 fast profile이며, native를 한 번 모두 만든 뒤
# group별로 init -> summarize -> freeze -> summarize -> multihop -> summarize 순서로 진행한다.
# 이렇게 하면 전체 종료를 기다리지 않아도 group 단위 중간 결과를 바로 볼 수 있다.
#
# 예:
#   ./run_ai502_transfer_all.sh --gpus 0,1,2,3
#   ./run_ai502_transfer_all.sh --gpus 0,1,2,3 --seeds 1,2,3,4,5 --hparams shared_1,shared_2
#   ./run_ai502_transfer_all.sh --gpus 0,1 --datasets beauty,retail_rocket --dry-run

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GPUS="0,1,2,3"
SEEDS="1,2,3"
DATASETS="beauty,foursquare,KuaiRecLargeStrictPosV2_0.2,lastfm0.03,movielens1m,retail_rocket"
HPARAMS="shared_3,shared_4,shared_5,shared_6"
CLEAN_ARTIFACTS=1
DRY_RUN=0
EXTRA_ARGS=()

ARTIFACT_DIR="${SCRIPT_DIR}/artifacts"
GROUP1_PAIRS="lastfm_to_KuaiRec,foursquare_to_KuaiRec,beauty_to_retail_rocket,retail_rocket_to_beauty"
GROUP1_TRIPLETS="beauty_to_retail_rocket_to_KuaiRec"
GROUP2_PAIRS="KuaiRec_to_foursquare,KuaiRec_to_lastfm,KuaiRec_to_movielens1m,lastfm_to_movielens1m"
GROUP2_TRIPLETS="foursquare_to_KuaiRec_to_lastfm"
GROUP3_PAIRS="lastfm_to_foursquare,KuaiRec_to_beauty,retail_rocket_to_KuaiRec,beauty_to_lastfm"
GROUP3_TRIPLETS="lastfm_to_KuaiRec_to_movielens1m,KuaiRec_to_foursquare_to_retail_rocket"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus)
      GPUS="$2"
      shift 2
      ;;
    --seeds)
      SEEDS="$2"
      shift 2
      ;;
    --datasets)
      DATASETS="$2"
      shift 2
      ;;
    --hparams)
      HPARAMS="$2"
      shift 2
      ;;
    --keep-artifacts)
      CLEAN_ARTIFACTS=0
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      EXTRA_ARGS+=("$1")
      shift
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

cleanup_artifacts() {
  local targets=(
    "${ARTIFACT_DIR}/analysis"
    "${ARTIFACT_DIR}/checkpoints"
    "${ARTIFACT_DIR}/hyperopt_results"
    "${ARTIFACT_DIR}/logging"
    "${ARTIFACT_DIR}/logs"
    "${ARTIFACT_DIR}/manifests"
    "${ARTIFACT_DIR}/summaries"
  )

  if [[ "${CLEAN_ARTIFACTS}" != "1" ]]; then
    echo "[AI502] keep-artifacts=on: 기존 artifacts를 유지합니다."
    return
  fi

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[AI502] dry-run: 아래 artifacts를 비우는 순서만 확인합니다."
    printf '  - %s\n' "${targets[@]}"
    return
  fi

  echo "[AI502] removing previous artifacts"
  rm -rf "${targets[@]}"
  mkdir -p "${ARTIFACT_DIR}"
}

run_summary() {
  echo "[AI502] summarize current transfer outputs"
  "${PY_BIN}" summarize_ai502_transfer.py
}

run_group() {
  local group_name="$1"
  local pairs="$2"
  local triplets="$3"

  echo
  echo "[AI502] ===== ${group_name}: init ====="
  "${PY_BIN}" run_ai502_transfer.py \
    --phase init \
    --profile fast \
    --gpus "${GPUS}" \
    --seeds "${SEEDS}" \
    --datasets "${DATASETS}" \
    --hparams "${HPARAMS}" \
    --lr-mode fixed1 \
    --pairs "${pairs}" \
    "${EXTRA_ARGS[@]}"
  run_summary

  echo
  echo "[AI502] ===== ${group_name}: freeze ====="
  "${PY_BIN}" run_ai502_transfer.py \
    --phase freeze \
    --profile fast \
    --gpus "${GPUS}" \
    --seeds "${SEEDS}" \
    --datasets "${DATASETS}" \
    --hparams "${HPARAMS}" \
    --lr-mode fixed1 \
    --pairs "${pairs}" \
    "${EXTRA_ARGS[@]}"
  run_summary

  if [[ -n "${triplets}" ]]; then
    echo
    echo "[AI502] ===== ${group_name}: multihop ====="
    "${PY_BIN}" run_ai502_transfer.py \
      --phase multihop \
      --profile fast \
      --gpus "${GPUS}" \
      --seeds "${SEEDS}" \
      --datasets "${DATASETS}" \
      --hparams "${HPARAMS}" \
      --lr-mode fixed1 \
      --triplets "${triplets}" \
      "${EXTRA_ARGS[@]}"
    run_summary
  fi
}

cd "${SCRIPT_DIR}"
PY_BIN="${RUN_PYTHON_BIN:-${PYTHON_BIN:-/venv/FMoE/bin/python}}"
if [[ ! -x "${PY_BIN}" ]]; then
  PY_BIN="$(command -v python3)"
fi
export RUN_PYTHON_BIN="${PY_BIN}"
export PYTHONPATH="/workspace/FeaturedMoE/experiments:/workspace/FeaturedMoE${PYTHONPATH:+:${PYTHONPATH}}"
echo "[AI502] python=${PY_BIN}"
cleanup_artifacts

echo
echo "[AI502] ===== native checkpoint bank ====="
"${PY_BIN}" run_ai502_transfer.py \
  --phase native \
  --profile fast \
  --gpus "${GPUS}" \
  --seeds "${SEEDS}" \
  --datasets "${DATASETS}" \
  --hparams "${HPARAMS}" \
  --lr-mode fixed1 \
  "${EXTRA_ARGS[@]}"
run_summary

run_group "group1" "${GROUP1_PAIRS}" "${GROUP1_TRIPLETS}"
run_group "group2" "${GROUP2_PAIRS}" "${GROUP2_TRIPLETS}"
run_group "group3" "${GROUP3_PAIRS}" "${GROUP3_TRIPLETS}"
