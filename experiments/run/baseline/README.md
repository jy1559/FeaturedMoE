# Baseline Run Entrypoints

- `train_single.sh`: single train run.
- `tune_by_dataset.sh`: one dataset, multiple models.
- `tune_by_model.sh`: one model, multiple datasets.

Outputs:

- Logs: `experiments/run/artifacts/logs/baseline/*`
- Results(JSON): `experiments/run/artifacts/results/baseline/*`
- Timeline events: `experiments/run/artifacts/timeline/events.jsonl`

Examples:

```bash
bash experiments/run/baseline/train_single.sh --dataset movielens1m --model sasrec --gpu 0
bash experiments/run/baseline/tune_by_dataset.sh --dataset kuairec0.3 --models sasrec,gru4rec --gpus 0,1 --max-evals 25
bash experiments/run/baseline/tune_by_model.sh --model patt --datasets movielens1m,retail_rocket --gpus 0,1
```
