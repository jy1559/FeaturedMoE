#!/usr/bin/env bash
set -euo pipefail
cd /workspace/FeaturedMoE
/venv/FMoE/bin/python experiments/run/final_experiment/ablation/a08_behavior_cases.py "$@"
