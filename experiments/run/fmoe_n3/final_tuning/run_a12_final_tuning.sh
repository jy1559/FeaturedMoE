#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "${SCRIPT_DIR}/stage1_family_sweep.sh" "$@"
bash "${SCRIPT_DIR}/stage2_dataset_refine.sh" "$@"
bash "${SCRIPT_DIR}/stage3_local_polish.sh" "$@"
