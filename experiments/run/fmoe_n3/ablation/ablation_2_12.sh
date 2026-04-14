#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENTS_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

cd "${EXPERIMENTS_DIR}"
PYTHON_BIN="${RUN_PYTHON_BIN:-/venv/FMoE/bin/python}"

"${PYTHON_BIN}" run/fmoe_n3/ablation/ablation_2_12.py "$@"
