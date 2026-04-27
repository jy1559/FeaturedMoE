Paper-ready plotting values for `q2` to `q4`.

These CSVs are the only inputs used by the plotting notebooks:

- `q2_plot_values.csv`
- `q3_temporal_plot_values.csv`
- `q3_routing_org_plot_values.csv`
- `q4_portability_plot_values.csv`
- `q4_efficacy_plot_values.csv`

Each row is a single plotted bar/line point. Update the metric values here and rerun the notebook generator plus notebook execution to refresh the figures.

Minimal schema:

- `dataset`: dataset key used by the notebook
- `variant_label` or task-specific label column: plotted method / setting name
- `variant_order`: plotting order
- `status`: free-form run status note
- `best_valid_seen_mrr20`: optional validation-side reference value
- `test_ndcg20`: plotted `NDCG@20`
- `test_hit10`: plotted `HR@10`

Q4 files also include aliases used elsewhere:

- `test_ndcg_20`
- `test_hr10`

Raw export tables remain in `../` and are no longer read by the paper plotting notebooks.
