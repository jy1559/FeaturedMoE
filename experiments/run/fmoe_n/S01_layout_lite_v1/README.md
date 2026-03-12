# S01_layout_lite_v1

Aggressive architecture-control state for `FeaturedMoE_N`.

- default runner:
  - `phase_arch_probe.sh`
- default phase prefix:
  - `ARCH3`
- plan note:
  - `ARCH_LITE_PLAN.md`
- shared common entrypoint:
  - `../tune_hparam.sh`

Canonical artifact roots:

- `../../artifacts/logs/fmoe_n/s01_layout_lite_v1/`
- `../../artifacts/results/fmoe_n/normal/s01_layout_lite_v1/`
- `../../artifacts/results/fmoe_n/special/s01_layout_lite_v1/`
- `../../artifacts/inventory/fmoe_n/`

Dataset mode:

- one dataset per run
- supported:
  - `KuaiRecSmall0.1`
  - `lastfm0.03`

ARCH3 focus:

- plain anchor re-check on strong layouts
- `MoE-off` control via `dense_ffn`
- `pure_attention` control via `identity`
- `full rule` router branch
- `z-loss` and early-only gate entropy regularization
- group-aware router feature summaries (`mean`, `mean_std`)

Default budget:

- `28 combos / dataset`
- `7 waves`
- `max_evals=4`
- `epochs=100`
- `patience=10`

