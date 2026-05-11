#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=/dev/null
source "${RUN_DIR}/common/run_metadata.sh"

DATASET="movielens1m"
PHASE_PREFIX="P1GRID"
LIMIT="20"
ROOT=""

usage() {
  cat <<USAGE
Usage: $0 [--dataset movielens1m] [--phase-prefix P1GRID] [--limit 20] [--root <results_dir>]
USAGE
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2 ;;
    --phase-prefix) PHASE_PREFIX="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --root) ROOT="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

if [ -z "$ROOT" ]; then
  ROOT="$(run_results_dir fmoe)"
fi

python - <<'PY' "$DATASET" "$PHASE_PREFIX" "$LIMIT" "$ROOT"
from pathlib import Path
import json,sys

dataset=sys.argv[1].lower()
phase_prefix=sys.argv[2]
limit=int(sys.argv[3])
root=Path(sys.argv[4])

rows=[]
if root.exists():
    for p in root.glob("*.json"):
        try:
            d=json.load(open(p,'r',encoding='utf-8'))
        except Exception:
            continue
        ds=str(d.get("dataset","")).lower()
        if ds!=dataset:
            continue
        phase=str(d.get("run_phase",""))
        if not phase.startswith(phase_prefix):
            continue
        best=d.get("best_mrr@20")
        if best is None:
            continue
        bp=d.get("best_params") or {}
        fs=d.get("fixed_search") or {}
        cf=d.get("context_fixed") or {}
        layout = cf.get("arch_layout_catalog")
        if layout is None:
            layout = fs.get("arch_layout_catalog", fs.get("arch_layout_id"))
        rows.append({
            "best": float(best),
            "phase": phase,
            "file": p.name,
            "layout": layout,
            "schedule": {
                "enable": fs.get("fmoe_schedule_enable", cf.get("fmoe_schedule_enable")),
                "alpha_until": fs.get("alpha_warmup_until", cf.get("alpha_warmup_until")),
                "temp_until": fs.get("temperature_warmup_until", cf.get("temperature_warmup_until")),
                "topk": fs.get("moe_top_k", cf.get("moe_top_k")),
                "topk_warmup": fs.get("moe_top_k_warmup_until", cf.get("moe_top_k_warmup_until")),
            },
            "hparam": {
                "lr": bp.get("learning_rate"),
                "wd": bp.get("weight_decay"),
                "drop": bp.get("hidden_dropout_prob"),
                "bal": bp.get("balance_loss_lambda"),
            },
        })

rows.sort(key=lambda x: x["best"], reverse=True)
print(f"[REPORT] dataset={dataset} phase_prefix={phase_prefix} matches={len(rows)}")
for i,r in enumerate(rows[:limit], start=1):
    print(
        f"{i:2d}. best={r['best']:.4f} phase={r['phase']} "
        f"lr={r['hparam']['lr']} wd={r['hparam']['wd']} "
        f"drop={r['hparam']['drop']} bal={r['hparam']['bal']} "
        f"layout={r['layout']} sched={r['schedule']} file={r['file']}"
    )
PY
