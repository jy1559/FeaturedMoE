# Results Workspace Guide

This directory is the staging area for paper tables and figures.

The structure follows the paper order so that experiment output, plotting notebooks, and generated assets stay aligned with the manuscript without renaming files later.

## Naming Rules

- Main-paper assets use `01_` to `04_` prefixes.
- Appendix assets use `A01_` to `A05_` prefixes.
- Each section folder keeps its own CSV templates and one notebook that reads those CSVs.
- Generated figures are written to `generated_figures/` with the same section prefix.
- CSV files use long/tidy format by default so the same file can drive a table, a bar plot, or a sanity-check pivot.

## Common CSV Conventions

Most CSVs reuse these metadata fields when they make sense:

- `paper_section`
- `panel`
- `dataset`
- `variant_or_model`
- `metric`
- `cutoff`
- `value`
- `split`
- `selection_rule`
- `run_id`
- `source_path`
- `notes`

Keep the raw extraction path in `source_path` whenever possible. That makes later verification easier when a manuscript number changes.

## Shared Plotting

All notebooks import plotting code from `_shared/`.

- `_shared/paper_theme.py`: theme, palette, figure defaults
- `_shared/plot_builders.py`: grouped bars, lines, scatter, heatmaps, annotations
- `_shared/io_helpers.py`: CSV loading, demo-data fallback, figure export

If figure styling changes later, update `_shared/` first instead of editing every notebook.

## Paper-Order Sections

### 01 Main Overall

- Purpose: generate the compact main-paper overall comparison and sanity-check the main table before LaTeX insertion.
- Key outputs: main overall table preview, compact overall comparison figure for inspection.
- Required metrics: `HR@10`, `NDCG@10`, `MRR@20`.
- Grouping: dataset x model.
- CSVs:
  - `01_main_overall/01_main_overall.csv`
- Notebook:
  - `01_main_overall/01_main_overall.ipynb`
- Likely raw inputs:
  - `outputs/baseline2_model_topk_summary.csv`
  - `experiments/run/artifacts/results/`

### 02 Routing Control

- Purpose: compare shared FFN, hidden-only routing, mixed routing, and behavior-guided routing.
- Key outputs: main routing-control figure with a quality panel and a consistency panel.
- Required metrics:
  - quality panel: ranking metric, usually `MRR@20`
  - consistency panel: route similarity or consistency by similarity bucket
- CSVs:
  - `02_routing_control/02a_routing_control_quality.csv`
  - `02_routing_control/02b_routing_control_consistency.csv`
- Notebook:
  - `02_routing_control/02_routing_control.ipynb`
- Likely raw inputs:
  - routing diagnostics from `experiments/run/fmoe_n3/docs/data/phase10_13/`

### 03 Stage Structure

- Purpose: justify the staged RouteRec architecture.
- Key outputs: stage-removal panel, dense-vs-staged panel, wrapper/order panel.
- Required metrics: usually `MRR@20` or the same scalar validation metric used in the paper.
- CSVs:
  - `03_stage_structure/03a_stage_ablation.csv`
  - `03_stage_structure/03b_dense_vs_staged.csv`
  - `03_stage_structure/03c_wrapper_order.csv`
- Notebook:
  - `03_stage_structure/03_stage_structure.ipynb`
- Likely raw inputs:
  - stage progression CSVs under `outputs/`
  - architecture sweeps under experiment artifacts

### 04 Cue Ablation

- Purpose: show whether lightweight cues are enough and how much gain survives under weaker metadata.
- Key outputs: cue-family ablation panel and normalized retention panel.
- Required metrics: usually `MRR@20`, optional relative gain.
- CSVs:
  - `04_cue_ablation/04a_cue_ablation.csv`
  - `04_cue_ablation/04b_cue_retention.csv`
- Notebook:
  - `04_cue_ablation/04_cue_ablation.ipynb`
- Likely raw inputs:
  - ablation summaries under `outputs/`
  - ablation docs under `experiments/run/fmoe_n3/ablation/`

### A01 Full Results

- Purpose: rebuild the appendix full-cutoff results table from one canonical CSV.
- Key outputs: appendix full grid table preview and optional improvement heatmap for inspection.
- Required metrics: `HR`, `NDCG`, `MRR` at `k=5,10,20`.
- CSVs:
  - `A01_full_results/A01_full_results.csv`
- Notebook:
  - `A01_full_results/A01_full_results.ipynb`

### A02 Objective Variants

- Purpose: compare training objectives and regularizers.
- Key outputs: compact appendix table preview plus a multi-measure comparison plot.
- Required metrics: ranking quality, route consistency, stability.
- CSVs:
  - `A02_objective_variants/A02_objective_variants.csv`
- Notebook:
  - `A02_objective_variants/A02_objective_variants.ipynb`

### A03 Routing Diagnostics

- Purpose: inspect expert usage, entropy, consistency, and feature-bucket routing patterns.
- Key outputs: a four-panel appendix diagnostics figure.
- CSVs:
  - `A03_routing_diagnostics/A03a_expert_usage.csv`
  - `A03_routing_diagnostics/A03b_entropy_effective_experts.csv`
  - `A03_routing_diagnostics/A03c_stage_consistency.csv`
  - `A03_routing_diagnostics/A03d_feature_bucket_patterns.csv`
- Notebook:
  - `A03_routing_diagnostics/A03_routing_diagnostics.ipynb`
- Likely raw inputs:
  - routing CSVs in `experiments/run/fmoe_n3/docs/data/phase10_13/`

### A04 Behavior Slices

- Purpose: localize RouteRec gains to behavioral regimes.
- Key outputs: slice-wise ranking panel and gain/concentration panel.
- Recommended slices:
  - repeat-heavy prefixes
  - fast-tempo sessions
  - narrow-focus sessions
  - exploration-heavy prefixes
- CSVs:
  - `A04_behavior_slices/A04a_slice_metrics.csv`
  - `A04_behavior_slices/A04b_slice_gain_concentration.csv`
- Notebook:
  - `A04_behavior_slices/A04_behavior_slices.ipynb`

### A05 Transfer

- Purpose: summarize source-target transfer, low-resource adaptation, and transfer variants.
- Key outputs: transfer matrix preview, low-resource curves, and transfer-variant comparison figure.
- CSVs:
  - `A05_transfer/A05a_transfer_matrix.csv`
  - `A05_transfer/A05b_low_resource_transfer.csv`
  - `A05_transfer/A05c_transfer_variants.csv`
- Notebook:
  - `A05_transfer/A05_transfer.ipynb`
- Likely raw inputs:
  - `outputs/transfer_stageA_dest_compare_best_of_source.csv`
  - `outputs/transfer_stageA_dest_runs_metrics.csv`

## Recommended Workflow

1. Extract or aggregate experiment output into the matching section CSV.
2. Open the matching notebook and confirm the preview table looks correct.
3. Regenerate the figure into `generated_figures/`.
4. Copy the generated asset path into the LaTeX draft only after the preview matches the intended paper figure.

## Notes

- Empty CSVs are allowed. The notebooks fall back to demo data so plotting code can be tested before the final results arrive.
- When a paper figure changes later, keep the CSV schema stable if possible. That reduces downstream cleanup.
- Prefer updating the shared plotting helpers before touching notebook-local styling.