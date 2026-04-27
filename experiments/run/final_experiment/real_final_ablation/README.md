# real_final_ablation

Main-body RouteRec Q2~Q5 experiment stack for the real-final paper refresh.

## Scope

- `Q2`: routing source
- `Q3`: design justification
- `Q4`: reduced-cue portability
- `Q5`: learned routing behavior

This folder is intentionally separate from the older `final_experiment/ablation` stack. The older portability-heavy and broader appendix sweeps remain available there; this folder only covers the refreshed main-text storyline and the exports needed by `writing/260419_real_final_exp`.

## Entry Points

- `run_q2_q5_suite.sh`
- `run_q2_q5_suite.py`
- `q2_routing_control.py`
- `q3_stage_structure.py`
- `q4_portability.py`
- `q4_eval_feature_efficacy.py`
- `q4_experiment_guide.md`
- `q5_behavior_semantics.py`
- `export_q2_q5_bundle.py`

## Output Layout

- Logs: `experiments/run/artifacts/logs/real_final_ablation`
- Results: `experiments/run/artifacts/results/real_final_ablation`
- Notebook data: `writing/260419_real_final_exp/data`

## Expected Notebook Exports

- `q2_quality.csv`
- `q3_temporal_decomp.csv`
- `q3_routing_org.csv`
- `q4_portability_table.csv`
- `q4_feature_efficacy.csv`
- `q5_case_heatmap.csv`
- `q5_intervention_summary.csv`
- `q_suite_manifest.json`
- `q_suite_run_index.csv`

## Notes

- Base candidates default to `final_experiment/ablation/configs/base_candidates.csv`.
- Q4 uses a `1 + 3 + 3` design.
- Q4-A retrains `Full`, `No Group Cues`, `No Time Cues`, and `Portable Core`.
- Q4-B evaluates one trained `Full` checkpoint under `Intact`, `Zero All`, `Intra-Sequence Permute`, and `Cross-Sample Permute` cue conditions.
- Q5 keeps intervention outputs as supporting evidence, while the main-body export centers on representative case heatmaps.
