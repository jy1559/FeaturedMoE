#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GPU_LIST="0,1,2,3"
DRY_RUN="${DRY_RUN:-false}"
KUAI_MODE="${KUAI_MODE:-k20}"

# KuaiRec: 5 combos from P3 (4 structures + 1 strong variant)
K5_ONLY="P3S1_01,P3S2_01,P3S3_01,P3S4_01,P3S2_02"

# lastfm: 7 combos from core_28 (plain/dense/rule/token/scale mix)
L7_ONLY="P00,P01,D10,D11,R30,T50,X60"

usage() {
  cat <<USAGE
Usage: $0 [--gpus 0,1,2,3] [--dry-run] [--kuai-mode k20|k5]

Runs in sequence with existing defaults:
1) KuaiRecLargeStrictPosV2_0.2 on phase_3_20.sh with 20 combos (default)
2) lastfm0.03 on phase_core_28.sh with 7 combos

Only --gpus is required/expected. Other parameters follow each phase script defaults.

Notes:
- --kuai-mode k20: run full P3 20 combos (default)
- --kuai-mode k5 : run curated 5-combo quick pass
USAGE
}

while [ $# -gt 0 ]; do
  case "$1" in
    --gpus)
      GPU_LIST="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN="true"
      shift
      ;;
    --kuai-mode)
      KUAI_MODE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [ "${KUAI_MODE}" != "k20" ] && [ "${KUAI_MODE}" != "k5" ]; then
  echo "Invalid --kuai-mode: ${KUAI_MODE} (expected k20 or k5)" >&2
  usage >&2
  exit 1
fi

CMD_K=(
  "${SCRIPT_DIR}/phase_3_20.sh"
  --datasets "KuaiRecLargeStrictPosV2_0.2"
  --gpus "${GPU_LIST}"
)

if [ "${KUAI_MODE}" = "k5" ]; then
  CMD_K+=(--only "${K5_ONLY}")
fi

CMD_L=(
  "${SCRIPT_DIR}/phase_core_28.sh"
  --dataset "lastfm0.03"
  --gpus "${GPU_LIST}"
  --only "${L7_ONLY}"
)

if [ "${DRY_RUN}" = "true" ]; then
  CMD_K+=(--dry-run)
  CMD_L+=(--dry-run)
fi

if [ "${KUAI_MODE}" = "k5" ]; then
  echo "[Stage 1/2] KuaiRec P3 quick 5 combos"
else
  echo "[Stage 1/2] KuaiRec P3 full 20 combos"
fi
"${CMD_K[@]}"

echo "[Stage 2/2] lastfm core_28 7 combos"
"${CMD_L[@]}"

echo "[All Done] K5 + L7 completed"
