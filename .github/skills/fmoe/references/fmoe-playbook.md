# FMoE Playbook

## Current Priority
- Main track: fmoe_n3
- Main datasets: KuaiRecLargeStrictPosV2_0.2, lastfm0.03
- Main metric: MRR@20
- Baseline anchor: SASRec

## Standard Flow
1. Dry-run first.
```bash
bash experiments/run/fmoe_n3/phase_core_28.sh --dataset KuaiRecLargeStrictPosV2_0.2 --gpus 0,1 --dry-run
```

2. Execute main fmoe_n3 run.
```bash
bash experiments/run/fmoe_n3/phase_core_28.sh --dataset KuaiRecLargeStrictPosV2_0.2 --gpus 0,1,2,3 --seed-base 8300 --use-recommended-budget
```

3. Run on lastfm0.03.
```bash
bash experiments/run/fmoe_n3/phase_core_28.sh --dataset lastfm0.03 --gpus 0,1 --seed-base 9300 --use-recommended-budget
```

4. Run SASRec baseline for direct comparison.
```bash
bash experiments/run/baseline/tune_by_model.sh --model SASRec --datasets KuaiRecLargeStrictPosV2_0.2,lastfm0.03 --gpus 0,1 --phase P0 --max-evals 20 --tune-epochs 100 --tune-patience 10
```

5. Summarize and propose next trials.
```bash
python3 .codex/skills/fmoe/scripts/collect_results.py --repo-root /workspace/jy1559/FMoE --datasets KuaiRecLargeStrictPosV2_0.2,lastfm0.03 --metric mrr@20
python3 .codex/skills/fmoe/scripts/recommend_next.py --summary /workspace/jy1559/FMoE/experiments/run/artifacts/results/summary.csv --mode fmoe-first --topn 3
```

## Reading Outputs
- normal result JSON: overall best parameters and summary metrics
- special metrics JSON: difficult slice performance (short sessions, popularity slices)
- diag files: routing distribution/balance and stability signals
- feature_ablation fields: sensitivity to missing/shuffled feature inputs
