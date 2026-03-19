# P6 Clean Replot (Outlier-Removed + Top-k/Expert-Aware)

- outlier handling: x-axis IQR filter (1.5*IQR), so ~0.59 extreme point removed automatically.
- normalization: n_eff_norm = n_eff / active_cap
- active_cap rule: if moe_top_k>0 then active_cap=moe_top_k else active_cap=expert_count(=12 fallback).

Generated SVG:
- outputs/phase56_corr/p6_clean_macro_n_eff_norm_vs_valid.svg
- outputs/phase56_corr/p6_clean_micro_n_eff_norm_vs_valid.svg
- outputs/phase56_corr/p6_clean_macro_cv_usage_vs_valid.svg
- outputs/phase56_corr/p6_clean_macro_top1max_vs_valid.svg
- outputs/phase56_corr/p6_clean_macro_top1xcap_vs_valid.svg
- outputs/phase56_corr/p6_clean_macro_knn_js_vs_valid.svg