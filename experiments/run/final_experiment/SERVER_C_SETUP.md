# Server C Setup

This file explains the safest way to bring up a fresh `server_c` worker for the current `final_experiment` FMoE-only track.

## Recommendation

Use GitHub for the base repository, then copy a small set of local files from this server.

Why:

- `origin/server/20260417-snapshot` can be fetched and checked out safely.
- That remote branch does **not** contain [`experiments/run/final_experiment`](/workspace/FeaturedMoE/experiments/run/final_experiment).
- The current local [`hyperopt_tune.py`](/workspace/FeaturedMoE/experiments/hyperopt_tune.py) differs from `origin/server/20260417-snapshot` and includes the current retry, time-budget, and final-experiment fixes.
- `build_space_manifest.py` depends on local historical result trees (`artifacts/results/baseline_2`, `artifacts/results/fmoe_n4`), so a fresh server should normally **reuse the prebuilt manifest** instead of rebuilding it.

## What To Get From GitHub

Clone the repository and check out the branch you want as the base code snapshot.

Example:

```bash
git clone git@github.com:jy1559/FeaturedMoE.git /workspace/FeaturedMoE
cd /workspace/FeaturedMoE
git fetch origin --prune
git checkout server/20260417-snapshot
```

If you do not want to disturb an existing dirty tree, prefer a separate worktree:

```bash
git fetch origin --prune
git worktree add ../FeaturedMoE_serverc origin/server/20260417-snapshot
```

## What To Copy From This Server

Copy these files or directories from this server to the fresh server:

1. Whole final-experiment runner directory

```text
experiments/run/final_experiment/
```

2. Patched shared runtime

```text
experiments/hyperopt_tune.py
```

3. Environment file

```text
experiments/env/FMoE_env_server_c_20260418.yml
```

4. Dataset files required by `session_fixed + feature_added_v4`

```text
Datasets/processed/feature_added_v4/beauty/
Datasets/processed/feature_added_v4/foursquare/
Datasets/processed/feature_added_v4/KuaiRecLargeStrictPosV2_0.2/
Datasets/processed/feature_added_v4/lastfm0.03/
Datasets/processed/feature_added_v4/movielens1m/
Datasets/processed/feature_added_v4/retail_rocket/
```

Required files per dataset:

```text
<dataset>.train.inter
<dataset>.valid.inter
<dataset>.test.inter
```

The current validation check is implemented in [common.py](/workspace/FeaturedMoE/experiments/run/final_experiment/common.py:666).

## If You Want Server C To Resume Current FMoE Progress

If server C should continue from the work already done on this server, copy the current FMoE stage artifacts too.

Recommended copy set:

1. Stage manifest and summary

```text
experiments/run/artifacts/logs/final_experiment/stage1/manifest.json
experiments/run/artifacts/logs/final_experiment/stage1/summary.csv
```

2. FMoE stage logs

```text
experiments/run/artifacts/logs/final_experiment/stage1/<dataset>/featured_moe_n3/
experiments/run/artifacts/logs/final_experiment/special/stage1_broad_search/S1/<dataset>/FMoEN3/
```

3. FMoE stage result trees

```text
experiments/run/artifacts/results/final_experiment/normal/stage1_broad_search/S1/<dataset>/FMoEN3/
experiments/run/artifacts/results/final_experiment/special/stage1_broad_search/S1/<dataset>/FMoEN3/
```

This lets `--resume-from-logs` and later stage selection reuse what has already finished instead of restarting from zero.

Even if the copied `stage1/summary.csv` is incomplete, `run_server_c.sh stage1...` will rebuild missing summary entries by skipping completed runs from the copied logs and result JSON files.

## Minimal Rsync Plan

Replace `USER@SERVER_C` with your real target.

```bash
rsync -av /workspace/FeaturedMoE/experiments/run/final_experiment/ USER@SERVER_C:/workspace/FeaturedMoE/experiments/run/final_experiment/
rsync -av /workspace/FeaturedMoE/experiments/hyperopt_tune.py USER@SERVER_C:/workspace/FeaturedMoE/experiments/hyperopt_tune.py
rsync -av /workspace/FeaturedMoE/experiments/env/FMoE_env_server_c_20260418.yml USER@SERVER_C:/workspace/FeaturedMoE/experiments/env/FMoE_env_server_c_20260418.yml
rsync -av /workspace/FeaturedMoE/Datasets/processed/feature_added_v4/beauty USER@SERVER_C:/workspace/FeaturedMoE/Datasets/processed/feature_added_v4/
rsync -av /workspace/FeaturedMoE/Datasets/processed/feature_added_v4/foursquare USER@SERVER_C:/workspace/FeaturedMoE/Datasets/processed/feature_added_v4/
rsync -av /workspace/FeaturedMoE/Datasets/processed/feature_added_v4/KuaiRecLargeStrictPosV2_0.2 USER@SERVER_C:/workspace/FeaturedMoE/Datasets/processed/feature_added_v4/
rsync -av /workspace/FeaturedMoE/Datasets/processed/feature_added_v4/lastfm0.03 USER@SERVER_C:/workspace/FeaturedMoE/Datasets/processed/feature_added_v4/
rsync -av /workspace/FeaturedMoE/Datasets/processed/feature_added_v4/movielens1m USER@SERVER_C:/workspace/FeaturedMoE/Datasets/processed/feature_added_v4/
rsync -av /workspace/FeaturedMoE/Datasets/processed/feature_added_v4/retail_rocket USER@SERVER_C:/workspace/FeaturedMoE/Datasets/processed/feature_added_v4/
```

If you want resume support on server C, also copy:

```bash
rsync -av /workspace/FeaturedMoE/experiments/run/artifacts/logs/final_experiment/stage1/manifest.json USER@SERVER_C:/workspace/FeaturedMoE/experiments/run/artifacts/logs/final_experiment/stage1/manifest.json
rsync -av /workspace/FeaturedMoE/experiments/run/artifacts/logs/final_experiment/stage1/summary.csv USER@SERVER_C:/workspace/FeaturedMoE/experiments/run/artifacts/logs/final_experiment/stage1/summary.csv
rsync -av /workspace/FeaturedMoE/experiments/run/artifacts/logs/final_experiment/stage1/beauty/featured_moe_n3 USER@SERVER_C:/workspace/FeaturedMoE/experiments/run/artifacts/logs/final_experiment/stage1/beauty/
rsync -av /workspace/FeaturedMoE/experiments/run/artifacts/logs/final_experiment/stage1/foursquare/featured_moe_n3 USER@SERVER_C:/workspace/FeaturedMoE/experiments/run/artifacts/logs/final_experiment/stage1/foursquare/
rsync -av /workspace/FeaturedMoE/experiments/run/artifacts/logs/final_experiment/stage1/kuaireclargestrictposv2_0_2/featured_moe_n3 USER@SERVER_C:/workspace/FeaturedMoE/experiments/run/artifacts/logs/final_experiment/stage1/kuaireclargestrictposv2_0_2/
rsync -av /workspace/FeaturedMoE/experiments/run/artifacts/logs/final_experiment/stage1/lastfm0_03/featured_moe_n3 USER@SERVER_C:/workspace/FeaturedMoE/experiments/run/artifacts/logs/final_experiment/stage1/lastfm0_03/
rsync -av /workspace/FeaturedMoE/experiments/run/artifacts/logs/final_experiment/stage1/movielens1m/featured_moe_n3 USER@SERVER_C:/workspace/FeaturedMoE/experiments/run/artifacts/logs/final_experiment/stage1/movielens1m/
rsync -av /workspace/FeaturedMoE/experiments/run/artifacts/logs/final_experiment/stage1/retail_rocket/featured_moe_n3 USER@SERVER_C:/workspace/FeaturedMoE/experiments/run/artifacts/logs/final_experiment/stage1/retail_rocket/
rsync -av /workspace/FeaturedMoE/experiments/run/artifacts/logs/final_experiment/special/stage1_broad_search/S1/ USER@SERVER_C:/workspace/FeaturedMoE/experiments/run/artifacts/logs/final_experiment/special/stage1_broad_search/S1/
rsync -av /workspace/FeaturedMoE/experiments/run/artifacts/results/final_experiment/normal/stage1_broad_search/S1/ USER@SERVER_C:/workspace/FeaturedMoE/experiments/run/artifacts/results/final_experiment/normal/stage1_broad_search/S1/
rsync -av /workspace/FeaturedMoE/experiments/run/artifacts/results/final_experiment/special/stage1_broad_search/S1/ USER@SERVER_C:/workspace/FeaturedMoE/experiments/run/artifacts/results/final_experiment/special/stage1_broad_search/S1/
```

## Environment Install

Create the FMoE environment from:

- [FMoE_env_server_c_20260418.yml](/workspace/FeaturedMoE/experiments/env/FMoE_env_server_c_20260418.yml)

Example:

```bash
conda env create -f /workspace/FeaturedMoE/experiments/env/FMoE_env_server_c_20260418.yml
conda activate FMoE-server-c
python -c "import torch, recbole, lightgbm; print(torch.__version__, recbole.__version__, lightgbm.__version__)"
```

## Important: Do Not Rebuild The Manifest On A Fresh Server

Fresh server C usually does **not** need the full historical result trees.

Instead, copy these prebuilt files from this server:

```text
experiments/run/final_experiment/space_manifest.json
experiments/run/final_experiment/tuning_space.csv
experiments/run/final_experiment/server_split.json
```

Then run with:

```bash
export SKIP_BUILD_SPACE=1
```

The wrappers now support this:

- [run_server_a.sh](/workspace/FeaturedMoE/experiments/run/final_experiment/run_server_a.sh:13)
- [run_server_b.sh](/workspace/FeaturedMoE/experiments/run/final_experiment/run_server_b.sh:13)
- [run_server_c.sh](/workspace/FeaturedMoE/experiments/run/final_experiment/run_server_c.sh:14)

## Recommended Server C Commands

Fast/core FMoE stage 1:

```bash
cd /workspace/FeaturedMoE
conda activate FMoE-server-c
export RUN_PYTHON_BIN="$(which python)"
export SKIP_BUILD_SPACE=1
bash experiments/run/final_experiment/run_server_c.sh stage1-fast
```

Slow FMoE stage 1:

```bash
bash experiments/run/final_experiment/run_server_c.sh stage1-slow
```

Stage 2:

```bash
bash experiments/run/final_experiment/run_server_c.sh stage2-fast
bash experiments/run/final_experiment/run_server_c.sh stage2-slow
```

Stage 3:

```bash
bash experiments/run/final_experiment/run_server_c.sh stage3
```

One-shot pipeline:

```bash
bash experiments/run/final_experiment/run_server_c.sh pipeline
```

See the route-stage design notes in [SERVER_C_ROUTE_PLAN.md](/workspace/FeaturedMoE/experiments/run/final_experiment/SERVER_C_ROUTE_PLAN.md).

## Sanity Checks

1. Confirm dataset files exist:

```bash
ls /workspace/FeaturedMoE/Datasets/processed/feature_added_v4/foursquare/
```

2. Confirm the wrapper sees the right interpreter:

```bash
echo "$RUN_PYTHON_BIN"
```

3. Dry-run before the full launch:

```bash
SKIP_BUILD_SPACE=1 bash experiments/run/final_experiment/run_server_c.sh stage1-fast --dry-run
```
