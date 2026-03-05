# FeaturedMoE_HiR Run Scripts

`run/fmoe_hir` is the only active HiR track.

## MoE on/off isolation run (same total attention layers)

```bash
bash experiments/run/fmoe_hir/run_moe_onoff_hir.sh --dataset movielens1m --gpu 7 --log-wandb
```

4 phases (same attn budget, MoE on/off only):
- `P4HIR_SER_MOE_ON_L4`   : serial,  layout `1,1,1,1,0`
- `P4HIR_SER_MOE_OFF_L4`  : serial,  layout `4,-1,-1,-1,0`
- `P4HIR_PAR_MOE_ON_L2`   : parallel, layout `2,0,0,0,0`
- `P4HIR_PAR_MOE_OFF_L2`  : parallel, layout `2,-1,-1,-1,0`

Defaults:
- `search_profile=wide`
- `max_evals=20`, `tune_epochs=100`, `tune_patience=10`
- `train/eval batch=16384/16384`
- `expert_scale=4` (fixed by `tune_hparam_hir.sh`)

## One-shot 4-phase run

```bash
bash experiments/run/fmoe_hir/run_4phase_hir.sh --dataset movielens1m --gpu 7 --log-wandb
```

4 phases (serial/parallel x off/temp_mild):
- `P3HIR_SER_off`
- `P3HIR_SER_temp`
- `P3HIR_PAR_off`
- `P3HIR_PAR_temp`

Defaults:
- `expert_scale=4` (fixed)
- wandb project: `FMoE_hir`
- run group/log group: `fmoe_hir`
- serial layout catalog: `0,1,1,0,0;1,1,1,1,0`
- parallel layout catalog: `2,0,0,0,0;4,0,0,0,0`

## Single phase run

```bash
bash experiments/run/fmoe_hir/tune_hparam_hir.sh \
  --dataset movielens1m \
  --gpu 7 \
  --phase P3HIR_SER_off \
  --stage-merge-mode serial \
  --schedule-preset off \
  --layout-catalog '0,1,1,0,0;1,1,1,1,0' \
  --log-wandb
```

## Output paths (artifacts-first)

- Logs: `experiments/run/artifacts/logs/fmoe_hir/*`
- Results: `experiments/run/artifacts/results/fmoe_hir/*.json`
- Timeline: `experiments/run/artifacts/timeline/events.jsonl`
