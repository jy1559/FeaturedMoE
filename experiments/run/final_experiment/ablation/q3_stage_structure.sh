#!/usr/bin/env bash
set -euo pipefail
cd /workspace/FeaturedMoE
/venv/FMoE/bin/python experiments/run/final_experiment/ablation/q3_stage_structure.py "$@"
