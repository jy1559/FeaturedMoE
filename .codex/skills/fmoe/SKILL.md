---
name: fmoe
description: FeaturedMoE sequential-recommendation experiment skill for /workspace/FeaturedMoE. Use when working on the current FeaturedMoE codebase, especially to understand or run `experiments/run/fmoe_n3/*` and `experiments/run/baseline/*`, inspect `session_fixed` / `feature_added_v3` datasets, resume StageH baseline or FMoE_N3 final-wrapper experiments, summarize what past runs established, or avoid Hydra override mistakes in `hyperopt_tune.py` and run scripts.
---

# FMoE

Read this skill when the task touches the current FeaturedMoE repository. Default to the current focus, not the historical branches.

## Current Focus

- Treat `experiments/run/fmoe_n3/*` and `experiments/run/baseline/*` as the main active surfaces.
- Treat `fmoe_n3` as the paper model track and `baseline StageH` as the main baseline recovery track.
- Treat older families such as `fmoe`, `fmoe_hir`, earlier phase scripts, and legacy configs as secondary unless the user explicitly asks for them.
- Assume the user usually wants action on the current workspace, not abstract architecture discussion.

## Read First

- For project purpose and paper framing, read [references/current-state.md](references/current-state.md).
- For entry points, datasets, and which files matter first, read [references/repo-map.md](references/repo-map.md).
- For Hydra and CLI override traps, read [references/hydra-overrides.md](references/hydra-overrides.md).

## Working Rules

- Start from the active runner or summary file instead of scanning the whole repo.
- For experiment status questions, check `experiments/run/artifacts/logs/.../summary.csv` and only then inspect per-run logs.
- For `fmoe_n3`, verify the dataset path alias first if file-missing errors mention `/workspace/jy1559/FMoE`.
- For `baseline StageH`, prefer the shell wrapper plus `run_stageH_targeted_recovery.py`; do not assume `--underperform-screen` was the last active mode.
- For `fmoe_n3 A8/A10/A11/A12 wrapper`, prefer the wrapper sweep shell script and dataset-filtered resumes.
- When suggesting commands, bias toward `--resume-from-logs` and dataset narrowing.

## Fast Checks

- Dataset path check: confirm `Datasets/processed/feature_added_v3/<dataset>/<dataset>.train|valid|test.inter`.
- Current FMoE_N3 wrapper status: inspect `experiments/run/artifacts/logs/fmoe_n3/Final_all_datasets/<dataset>/summary.csv`.
- Current StageH status: inspect `experiments/run/artifacts/logs/baseline/StageH_TargetedRecovery_anchor2_core5/<dataset>/summary.csv`.
- If StageH skip behavior matters, remember the current code also skips by `(dataset, model, candidate_id, seed)` summary keys, not only by strict log end markers.

## Command Style

- Prefer dry-run once before large resumes if the user is unsure.
- When the user wants Slack notifications, use the local `run_with_slack_notify.sh` wrappers.
- Keep Slack titles short and put extra context into `--note`.

## Do Not Forget

- `baseline` and `fmoe_n3` both currently use `eval_mode=session_fixed` and `feature_mode=full_v3`.
- `basic` datasets are not required for these two active tracks.
- `lastfm0.03` is the active LastFM dataset name; avoid drifting back to `lastfm0.3`.
- The safest default Python is `/venv/FMoE/bin/python` if it exists.
