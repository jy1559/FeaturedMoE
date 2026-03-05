# Run System

## Track boundaries

- `baseline`: `experiments/run/baseline/*` only
- `fmoe`: `experiments/run/fmoe/*` only
- `fmoe_hir`: `experiments/run/fmoe_hir/*` only
- common helpers: `experiments/run/common/*`

HiR scripts are not part of `run/fmoe`.

## Active outputs (artifacts-first)

- Logs: `experiments/run/artifacts/logs/<track>/<axis>/<phase>/<dataset>/<model>/*.log`
- Results: `experiments/run/artifacts/results/<track>/*.json`
- Timeline: `experiments/run/artifacts/timeline/events.jsonl`
- Dashboard: `experiments/run/artifacts/timeline/dashboard.md`
- Inventory: `experiments/run/artifacts/inventory/*`

Legacy folders (`experiments/run/log`, `experiments/run/hyperopt_results`) are kept for read fallback only.

## Quarantine policy

- Curated quarantine root for this reorg:
  `experiments/_quarantine/정리_260305/*`
- Category buckets:
  - `10_run_혼합스크립트`
  - `20_run_기존로그`
  - `21_run_기존결과`
  - `30_legacy_log`
  - `40_legacy_tensorboard`
  - `50_legacy_wandb`
  - `60_빈파일_깨진결과`
  - `70_구문서_노트북`
- Restore index files:
  `experiments/_quarantine/정리_260305/00_index/*`
