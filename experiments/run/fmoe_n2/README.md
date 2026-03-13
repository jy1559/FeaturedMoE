# FMoE_N2

Forked `FeaturedMoE_N2` track for feature-heavy router experiments.

- model aliases: `FeaturedMoE_N2`, `featured_moe_n2`, `featuredmoe_n2`
- canonical track name: `fmoe_n2`
- shared tune entrypoint: [tune_hparam.sh](/workspace/jy1559/FMoE/experiments/run/fmoe_n2/tune_hparam.sh)
- phase summary updater: [update_phase_summary.py](/workspace/jy1559/FMoE/experiments/run/fmoe_n2/update_phase_summary.py)
- default full-run dataset: `KuaiRecLargeStrictPosV2_0.2`
- smoke / transfer dataset: `lastfm0.03`

Artifacts stay under `run/artifacts`:

- logs: `experiments/run/artifacts/logs/fmoe_n2/<STATE>/<PHASE>/<DATASET>/`
- results(normal): `experiments/run/artifacts/results/fmoe_n2/normal/<STATE>/<PHASE>/<DATASET>/<MODEL>/`
- results(special): `experiments/run/artifacts/results/fmoe_n2/special/<STATE>/<PHASE>/<DATASET>/<MODEL>/`
- inventory: `experiments/run/artifacts/inventory/fmoe_n2/`

State workspaces:

- [S00_router_feature_heavy_v1](/workspace/jy1559/FMoE/experiments/run/fmoe_n2/S00_router_feature_heavy_v1)
