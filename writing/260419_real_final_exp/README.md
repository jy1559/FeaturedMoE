# 260419 Real-Final Experiment Figures

Fresh notebook package for the real-final RouteRec Q2~Q5 main-body figures.

## Files

- `02_q2_routing_control.ipynb`
- `03_q3_design_justification.ipynb`
- `04_q4_efficiency.ipynb`
- `05_q5_behavior_semantics.ipynb`
- `real_final_viz_helpers.py`

## Expected Data

These notebooks read from `writing/260419_real_final_exp/data`:

- `q2_quality.csv`
- `q2_routing_profile.csv`
- `q3_temporal_decomp.csv`
- `q3_routing_org.csv`
- `q4_efficiency_table.csv`
- `q5_case_heatmap.csv`
- `q5_intervention_summary.csv`

Populate the folder by running:

```bash
/venv/FMoE/bin/python experiments/run/final_experiment/real_final_ablation/export_q2_q5_bundle.py
```

## Notes

- The notebooks are intentionally lightweight and paper-slot oriented.
- Each notebook starts by loading the exported CSV and previewing the schema before plotting.
- The plotting code is designed to be easy to modify once the final numbers are available.
