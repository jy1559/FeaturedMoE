# top1_over_uniform SVG fix (active-aware)
- Recomputed x-axis metric as mean(top1_max_frac * n_active) across stages.
- n_active uses count of experts with usage_sum > 0 in diag_best_valid stage metrics.
- This makes the uniform baseline depend on active experts, avoiding <1 anomalies under old mixed-scope normalization.
- Updated SVG: /workspace/jy1559/FMoE/outputs/phase56_corr/p56_scatter_output_top1_over_uniform_mean_vs_best_valid_mrr20.svg
- Updated table copy: /workspace/jy1559/FMoE/outputs/phase56_corr/p56_balance_specialization_filtered_activeaware.csv
