# 260419 Real-Final Experiment Figures

Fresh notebook package for the real-final RouteRec Q2~Q5 main-body figures.

## Files

- `02_q2_routing_control.ipynb`
- `03_q3_design_justification.ipynb`
- `04_q4_portability.ipynb`
- `05_q5_behavior_semantics.ipynb`
- `real_final_viz_helpers.py`

Legacy notebook:

- `legacy/04_q4_efficiency.ipynb`

## Expected Data

These notebooks read from `writing/260419_real_final_exp/data`:

- `q2_quality.csv`
- `q3_temporal_decomp.csv`
- `q3_routing_org.csv`
- `q4_portability_table.csv`
- `q4_feature_efficacy.csv`
- `q5_case_heatmap.csv`
- `q5_intervention_summary.csv`

Q4 also has demo fallback CSVs for layout checks before real runs finish:

- `q4_portability_demo.csv`
- `q4_feature_efficacy_demo.csv`

Populate the folder by running:

```bash
/venv/FMoE/bin/python experiments/run/final_experiment/real_final_ablation/export_q2_q5_bundle.py
```

## Post-Run Pipeline

For the full post-run flow after Q2/Q3/Q5 experiments finish, use:

- [pipeline/Q2_Q3_Q5_POSTRUN_CHECKLIST.md](/workspace/FeaturedMoE/writing/260419_real_final_exp/pipeline/Q2_Q3_Q5_POSTRUN_CHECKLIST.md)
- [pipeline/postrun_q2_q5_pipeline.py](/workspace/FeaturedMoE/writing/260419_real_final_exp/pipeline/postrun_q2_q5_pipeline.py)

Recommended command:

```bash
cd /workspace/FeaturedMoE
/venv/FMoE/bin/python writing/260419_real_final_exp/pipeline/postrun_q2_q5_pipeline.py
```

## Notes

- The notebooks are intentionally lightweight and paper-slot oriented.
- Each notebook starts by loading the exported CSV and previewing the schema before plotting.
- The plotting code is designed to be easy to modify once the final numbers are available.
- `04_q4_portability.ipynb` prefers real exports when completed rows exist and falls back to demo CSVs when the current export is still empty, planned-only, or incomplete.
