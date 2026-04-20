# 260419 Real-Final Appendix Figures

Appendix notebook package for the RouteRec A→K evidence flow.

## Files

- `A01_appendix_full_results.ipynb`
- `A02_appendix_structural_ablation.ipynb`
- `A03_appendix_sparse_and_diagnostics.ipynb`
- `A04_appendix_behavior_and_bins.ipynb`
- `A05_appendix_interventions_and_cases.ipynb`
- `A06_appendix_optional_transfer.ipynb`
- `appendix_viz_helpers.py`

## Expected Data

These notebooks read from `writing/260419_real_final_exp/appendix/data`:

- `appendix_dataset_stats.csv`
- `appendix_full_results_long.csv`
- `appendix_structural_variants.csv`
- `appendix_sparse_tradeoff.csv`
- `appendix_sparse_diagnostics.csv`
- `appendix_objective_variants.csv`
- `appendix_cost_summary.csv`
- `appendix_routing_diagnostics.csv`
- `appendix_special_bins.csv`
- `appendix_behavior_slice_quality.csv`
- `appendix_behavior_slice_profiles.csv`
- `appendix_case_routing_profile.csv`
- `appendix_intervention_summary.csv`
- `appendix_transfer_summary.csv`
- `appendix_manifest.json`

Populate the folder by running:

```bash
/venv/FMoE/bin/python experiments/run/final_experiment/real_final_ablation/appendix/export_appendix_bundle.py
```
