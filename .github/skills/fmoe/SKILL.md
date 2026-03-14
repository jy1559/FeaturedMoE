---
name: fmoe
description: 'FMoE_N3-first Featured MoE SeqRec workflow for this repository. Use when running or tuning fmoe_n3 experiments, prioritizing KuaiRecLargeStrictPosV2_0.2 and lastfm0.03, comparing against SASRec baseline, interpreting special/diag/feature_ablation outputs, diagnosing OOM or missing results, and recommending next experiment candidates.'
argument-hint: 'Task for FMoE workflow (run, summarize, diagnose, or recommend)'
user-invocable: true
---

# FMoE Skill

## When To Use
- Run the main FMoE_N3 track for KuaiRecLargeStrictPosV2_0.2 and lastfm0.03.
- Compare FMoE against SASRec baseline (FMoE is SASRec-based with MoE extension).
- Collect experiment summaries centered on MRR@20.
- Interpret special metrics for difficult slices (for example, short sessions / low-pop groups).
- Interpret diag outputs for routing dispersion, expert balance, and training stability.
- Interpret feature_ablation to quantify feature contribution and robustness.

## Quick Start
1. fmoe_n3 dry-run validation (current primary track).
```bash
bash experiments/run/fmoe_n3/phase_core_28.sh --dataset KuaiRecLargeStrictPosV2_0.2 --gpus 0,1 --dry-run
```
2. Main run on KuaiRecLargeStrictPosV2_0.2.
```bash
bash experiments/run/fmoe_n3/phase_core_28.sh --dataset KuaiRecLargeStrictPosV2_0.2 --gpus 0,1,2,3 --seed-base 8300 --use-recommended-budget
```
3. Main run on lastfm0.03.
```bash
bash experiments/run/fmoe_n3/phase_core_28.sh --dataset lastfm0.03 --gpus 0,1 --seed-base 9300 --use-recommended-budget
```
4. SASRec baseline comparison run.
```bash
bash experiments/run/baseline/tune_by_model.sh --model SASRec --datasets KuaiRecLargeStrictPosV2_0.2,lastfm0.03 --gpus 0,1 --phase P0 --max-evals 20 --tune-epochs 100 --tune-patience 10
```
5. Focused summary and next-trial recommendation.
```bash
python3 .codex/skills/fmoe/scripts/collect_results.py --repo-root /workspace/jy1559/FMoE --datasets KuaiRecLargeStrictPosV2_0.2,lastfm0.03 --metric mrr@20
python3 .codex/skills/fmoe/scripts/recommend_next.py --summary /workspace/jy1559/FMoE/experiments/run/artifacts/results/summary.csv --mode fmoe-first --topn 3
```

## Operating Policy
- Keep FMoE_N3 as primary track.
- Use KuaiRecLargeStrictPosV2_0.2 and lastfm0.03 as top-priority datasets.
- Always include SASRec baseline comparison for claims.
- Keep default metric fixed to MRR@20.
- Use single best score for acceptance unless explicitly asked for reproducibility-focused criteria.

## Metric Interpretation
- special metrics: slice-wise quality under difficult conditions (for example short sessions, low-pop targets).
- diag metrics/files: routing distribution, expert usage balance, collapse risk, and auxiliary balance-loss related stability signals.
- feature_ablation: delta when all/some features are removed or shuffled.

## Current Paths
- FMoE_N3 results root: experiments/run/artifacts/results/fmoe_n3
- FMoE_N3 special metrics root: experiments/run/artifacts/results/fmoe_n3/special
- FMoE_N3 diag root: experiments/run/artifacts/results/fmoe_n3/diag
- Baseline results root: experiments/run/artifacts/results/baseline

## Hydra Override Notes
- For standalone tune files without root seed, prefer `++seed=42`.
- For dotted search-space keys, use dict merge format:
  - `++search={rule_router.variant:[teacher_gls]}`

## Resources
- Playbook: [fmoe-playbook](./references/fmoe-playbook.md)
- HiR and extension notes: [hir-and-arch-extension](./references/hir-and-arch-extension.md)
- Repo map: [repo-map](./references/repo-map.md)
- Troubleshooting: [troubleshooting](./references/troubleshooting.md)
