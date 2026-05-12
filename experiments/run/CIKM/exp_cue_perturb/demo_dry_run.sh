#!/usr/bin/env bash
# Demo dry-run: GPU 없이 명령어/조건 확인.
# Usage: bash demo_dry_run.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "══════════════════════════════════════════════════════"
echo "  CUE PERTURBATION ABLATION — Dry Run Demo"
echo "══════════════════════════════════════════════════════"
echo ""

echo "── Group A: eval-only conditions (eval_perturb.py) ────"
echo ""
echo "  # P0 checkpoint 있을 때:"
echo "  python eval_perturb.py --gpu 0 --datasets KuaiRec foursquare"
echo ""
echo "  # P0 checkpoint 없을 때 (자동 학습):"
echo "  python eval_perturb.py --gpu 0 --auto-p0"
echo ""
echo "  # checkpoint 직접 지정:"
echo "  python eval_perturb.py --gpu 0 --checkpoint /path/to/best.pth --intact-mrr20 0.1234"
echo ""
python - <<'PYEOF'
from eval_perturb import EVAL_CONDITIONS
print("  조건 목록 (9개):")
for cond, cfg in EVAL_CONDITIONS.items():
    mode   = cfg["feature_perturb_mode"]
    apply  = cfg["feature_perturb_apply"]
    family = cfg.get("feature_perturb_family", "")
    fam_str = f"  family={family}" if family else ""
    print(f"    {cond:<28} mode={mode:<18} apply={apply}{fam_str}")
PYEOF

echo ""
echo "── Group B: train-time conditions (train_perturb.py) ──"
echo ""
echo "  python train_perturb.py --gpus 0 1"
echo "  python train_perturb.py --gpus 0 --conditions hidden_only both_zero  # 핵심만"
echo ""
python - <<'PYEOF'
from train_perturb import TRAIN_CONDITIONS, build_train_cmd

FAKE = {"learning_rate": 1e-3, "weight_decay": 1e-4}

print("  조건 목록 (5개):")
for cond, cfg in TRAIN_CONDITIONS.items():
    new_arch = "★ 새 구조" if cfg["new_arch"] else "  동일 구조"
    print(f"    [{new_arch}] {cond:<18}  {cfg['description']}")

print("")
print("  생성 명령 예시 (KuaiRec, fake lr=1e-3):")
for cond, cfg in list(TRAIN_CONDITIONS.items())[:2]:
    cmd = build_train_cmd(
        dataset="KuaiRec", condition=cond,
        condition_cfg=cfg, gpu_id="0", best_params=FAKE,
    )
    preview = " ".join(cmd[2:8])
    print(f"    [{cond}] {preview} ...")
    print(f"      overrides: {cfg['overrides']}")
PYEOF

echo ""
echo "── 결과 취합 ───────────────────────────────────────────"
echo "  python collect_results.py"
echo "  → results/cue_perturb_summary.csv"
echo "    columns: condition | KuaiRec_mrr20 | HR@1/10 | NDCG@1/10 | Δ | ..."
echo ""
echo "── 전체 실행 ───────────────────────────────────────────"
echo "  bash run_all_perturb.sh 0 1           # GPU 0: eval+KuaiRec, GPU 1: foursquare"
echo "  bash run_all_perturb.sh 0 --auto-p0   # P0 없으면 자동 학습"
echo ""
echo "── 예상 소요 시간 ──────────────────────────────────────"
echo "  eval_perturb:  KuaiRec ~1h, foursquare ~20min (조건당 ~5-7min)"
echo "  train_perturb: KuaiRec 조건당 ~5-6h, foursquare 조건당 ~45min"
echo "  전체 (GPU 2개): ~28-32h  (KuaiRec train × 5 조건이 bottleneck)"
echo "  최소 set: hidden_only + both_zero + eval_zero + eval_shuffle → ~13-15h"
echo ""
echo "══════════════════════════════════════════════════════"
echo "  Demo 완료. 실제 실행: bash run_all_perturb.sh <GPU_IDs>"
echo "══════════════════════════════════════════════════════"
