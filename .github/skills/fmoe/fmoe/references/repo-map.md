# FMoE Repo Map

## Root Layout
- Repo root: `/workspace/jy1559/FMoE`
- Data root: `/workspace/jy1559/FMoE/Datasets`
- Experiment root: `/workspace/jy1559/FMoE/experiments`
- Skill root: `/workspace/jy1559/FMoE/.codex/skills/fmoe`

## Dataset Paths
- Raw datasets: `/workspace/jy1559/FMoE/Datasets/raw`
- Processed (basic): `/workspace/jy1559/FMoE/Datasets/processed/basic`
- Processed (feature v2): `/workspace/jy1559/FMoE/Datasets/processed/feature_added_v2`

현재 `feature_added_v2` 데이터셋(6종):
- `movielens1m`
- `retail_rocket`
- `amazon_beauty`
- `foursquare`
- `KuaiRec0.3`
- `lastfm0.3`

## Model and Config Paths
- Model configs: `/workspace/jy1559/FMoE/experiments/configs/model`
- FMoE config: `featured_moe.yaml`, `featured_moe_tune.yaml`
- HiR config: `featured_moe_hir.yaml`, `featured_moe_hir_tune.yaml`
- Base config: `/workspace/jy1559/FMoE/experiments/configs/config.yaml`

## Run Entrypoints
- Baseline:
  - `/workspace/jy1559/FMoE/experiments/run/baseline/train_single.sh`
  - `/workspace/jy1559/FMoE/experiments/run/baseline/tune_by_dataset.sh`
- FMoE:
  - `/workspace/jy1559/FMoE/experiments/run/fmoe/pipeline_ml1_rr.sh`
  - `/workspace/jy1559/FMoE/experiments/run/fmoe/train_single.sh`
  - `/workspace/jy1559/FMoE/experiments/run/fmoe/tune_hparam.sh`
  - `/workspace/jy1559/FMoE/experiments/run/fmoe/tune_layout.sh`
  - `/workspace/jy1559/FMoE/experiments/run/fmoe/tune_schedule.sh`
- HiR:
  - `/workspace/jy1559/FMoE/experiments/run/fmoe_hir/tune_hparam_hir.sh`
  - `/workspace/jy1559/FMoE/experiments/run/fmoe_hir/run_4phase_hir.sh`

## Log and Result Paths
- Logs root: `/workspace/jy1559/FMoE/experiments/run/artifacts/logs`
- JSON results root: `/workspace/jy1559/FMoE/experiments/run/artifacts/results`
- Grouped results:
  - `/workspace/jy1559/FMoE/experiments/run/artifacts/results/fmoe`
  - `/workspace/jy1559/FMoE/experiments/run/artifacts/results/fmoe_hir`
  - `/workspace/jy1559/FMoE/experiments/run/artifacts/results/baseline`

주의:
- 과거 결과/로그는 `experiments/_quarantine/정리_260305/*`로 분류 이관되었다.
- 결과 수집 시 artifacts 경로 우선, legacy 경로 fallback 스캔을 수행한다.
