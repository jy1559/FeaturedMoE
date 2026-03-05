# FMoE Run Entrypoints

`run/fmoe` is for **FeaturedMoE only**.
HiR experiments must use `experiments/run/fmoe_hir/*`.

## Scripts

- `train_single.sh`: fixed-config train (P0 anchor)
- `tune_hparam.sh`: hparam axis (`learning_rate`, `weight_decay`, `hidden_dropout_prob`, `balance_loss_lambda`)
- `tune_layout.sh`: layout axis (`arch_layout_id`)
- `tune_schedule.sh`: schedule axis (`alpha`, `temp`, `topk`, `combined`)
- `grid_hparam_ml1.sh`: ML1 narrow-grid screening
- `report_grid_results.sh`: summarize by `run_phase` prefix
- `pipeline_ml1_rr.sh`: ML1 -> RetailRocket P0~P4 pipeline

## Output paths (artifacts-first)

- Logs: `experiments/run/artifacts/logs/fmoe/*`
- Results: `experiments/run/artifacts/results/fmoe/*.json`
- Timeline: `experiments/run/artifacts/timeline/events.jsonl`

Legacy path fallback:

- `experiments/run/log/fmoe/*`
- `experiments/run/hyperopt_results/fmoe/*.json`

## Examples

```bash
bash experiments/run/fmoe/tune_hparam.sh --dataset movielens1m --layout_id 7 --schedule off --gpu 0
bash experiments/run/fmoe/tune_layout.sh --dataset movielens1m --parent_result experiments/run/artifacts/results/fmoe/<p1>.json --gpu 0
bash experiments/run/fmoe/tune_schedule.sh --dataset retail_rocket --parent_result experiments/run/artifacts/results/fmoe/<p3>.json --mode alpha --gpu 1
bash experiments/run/fmoe/pipeline_ml1_rr.sh --datasets movielens1m,retail_rocket --gpus 0,1
```
