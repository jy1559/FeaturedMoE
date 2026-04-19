# real_final_ablation appendix

Appendix-oriented RouteRec experiment stack for the `260419_real_final_exp` paper refresh.

## Why This Folder Exists

- `../real_final_ablation` covers the refreshed main-body Q2~Q5 story.
- This folder widens that runner style into the appendix A→K evidence flow.
- It keeps the same CLI conventions and resume behavior, but writes to appendix-only artifact namespaces.

## Entry Points

- `full_results_export.py`
- `special_bins.py`
- `structural_ablation.py`
- `sparse_routing.py`
- `objective_variants.py`
- `cost_summary.py`
- `routing_diagnostics.py`
- `behavior_slices.py`
- `targeted_interventions.py`
- `optional_transfer.py`
- `run_appendix_suite.py`
- `export_appendix_bundle.py`

Each `.py` has a matching `.sh` wrapper that uses `/venv/FMoE/bin/python`.

## Artifact Layout

- Logs: `experiments/run/artifacts/logs/real_final_ablation/appendix/...`
- Results: `experiments/run/artifacts/results/real_final_ablation/appendix/...`
- Notebook-ready bundle: `writing/260419_real_final_exp/appendix/data/...`

The appendix suite namespaces its manifests, summary CSVs, case-eval exports, and intervention manifests so they do not collide with main-body runs.

## Design Notes

- Training and resume orchestration come from `final_experiment/real_final_ablation/common.py`.
- Sparse-routing, case-eval, and intervention-heavy appendix logic borrows settings and postprocess ideas from `final_experiment/ablation`.
- Optional transfer remains intentionally lightweight; it is scaffolded so the appendix can reserve the slot without blocking the higher-priority sections.

## Typical Run

```bash
/venv/FMoE/bin/python experiments/run/final_experiment/real_final_ablation/appendix/run_appendix_suite.py \
  --datasets KuaiRecLargeStrictPosV2_0.2,beauty \
  --top-k-configs 2 \
  --seeds 1,2 \
  --gpus 0,1
```

Then export the notebook bundle with:

```bash
/venv/FMoE/bin/python experiments/run/final_experiment/real_final_ablation/appendix/export_appendix_bundle.py
```
