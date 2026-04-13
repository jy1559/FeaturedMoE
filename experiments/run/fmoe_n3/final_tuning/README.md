# A12 Final Tuning

Hierarchical final tuning stack for `A12_ATTN_MICRO_BEFORE_NO_BIAS_ALL_W5` on `session_fixed`.

Stages:
- `stage1_family_sweep`: broad family search with discrete-only knobs and random search
- `stage2_dataset_refine`: winner-family refinement with TPE
- `stage3_local_polish`: final local polish with TPE

Main entrypoints:
- `run_a12_final_tuning.sh`
- `run_a12_final_tuning_with_slack.sh`

Artifacts:
- logs: `experiments/run/artifacts/logs/fmoe_n3/Final_tuning_A12/`
- manifests:
  - `stage1_manifest.json`
  - `stage2_manifest.json`
  - `stage3_manifest.json`

Typical usage:

```bash
bash experiments/run/fmoe_n3/final_tuning/run_a12_final_tuning_with_slack.sh
```

Dry-run:

```bash
bash experiments/run/fmoe_n3/final_tuning/stage1_family_sweep.sh --dry-run
```
