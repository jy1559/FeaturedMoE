# S00_base_v1

Baseline state for the current `FeaturedMoE_N` P0 / P0.5 rerun line.

- purpose: keep the current simple-flat baseline track isolated from later architectural changes
- canonical runners:
  - `phase_p0_anchor.sh`
  - `phase_p0_5_lr_narrow.sh`
- plan note: `P0_P05_PLAN.md`
- shared common entrypoint:
  - `../tune_hparam.sh`
- canonical artifact roots:
  - `../../artifacts/logs/fmoe_n/s00_base_v1/`
  - `../../artifacts/results/fmoe_n/normal/s00_base_v1/`
  - `../../artifacts/results/fmoe_n/special/s00_base_v1/`
  - `../../artifacts/inventory/fmoe_n/`

This state is the heavy/simple anchor before layout-light experiments.
