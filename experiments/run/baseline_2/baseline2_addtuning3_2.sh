#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER_PY="${SCRIPT_DIR}/baseline2_addtuning3_2.py"

resolve_python_bin() {
  if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    printf '%s\n' "${CONDA_PREFIX}/bin/python"
    return 0
  fi
  if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
    printf '%s\n' "${VIRTUAL_ENV}/bin/python"
    return 0
  fi
  if [[ -n "${PYTHON_BIN:-}" && -x "${PYTHON_BIN}" ]]; then
    printf '%s\n' "${PYTHON_BIN}"
    return 0
  fi
  if command -v python >/dev/null 2>&1; then
    command -v python
    return 0
  fi
  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return 0
  fi
  return 1
}

PYTHON_BIN="$(resolve_python_bin || true)"
if [[ -z "${PYTHON_BIN}" ]]; then
  echo "No usable python interpreter found. Activate the target conda env first." >&2
  exit 1
fi

GPUS="${GPUS:-0}"
EXTRA_ARGS=()
HAS_GPU_FLAG=0

if [[ $# -gt 0 ]]; then
  EXTRA_ARGS=("$@")
fi

for arg in "${EXTRA_ARGS[@]}"; do
  if [[ "${arg}" == "--gpus" ]]; then
    HAS_GPU_FLAG=1
  fi
done

echo "[baseline2_addtuning3_2.sh] python=${PYTHON_BIN}" >&2

RUN_ARGS=()

if [[ "${HAS_GPU_FLAG}" != "1" ]]; then
  RUN_ARGS+=(--gpus "${GPUS}")
fi

RUN_ARGS+=("${EXTRA_ARGS[@]}")

exec "${PYTHON_BIN}" "${RUNNER_PY}" "${RUN_ARGS[@]}"