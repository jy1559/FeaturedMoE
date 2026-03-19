# P6 Clean Replot v2 (Top-k contamination aware)

- removed extreme x outliers with IQR filter (1.5*IQR).
- primary normalization updated to n_eff / expert_count (stable across top-k settings).
- added top1_over_uniform = top1_max_frac * expert_count.
- added stratified plots for moe_top_k=0 vs moe_top_k>0 to reduce top-k contamination.

Generated SVG:
- outputs/phase56_corr/p6_clean_v2_macro_n_eff_expnorm_vs_valid.svg
- outputs/phase56_corr/p6_clean_v2_macro_cv_usage_vs_valid.svg
- outputs/phase56_corr/p6_clean_v2_macro_top1_over_uniform_vs_valid.svg
- outputs/phase56_corr/p6_clean_v2_macro_knn_js_vs_valid.svg
- outputs/phase56_corr/p6_clean_v2_k0_macro_n_eff_expnorm_vs_valid.svg
- outputs/phase56_corr/p6_clean_v2_k0_macro_top1_over_uniform_vs_valid.svg
- outputs/phase56_corr/p6_clean_v2_kpos_macro_n_eff_expnorm_vs_valid.svg
- outputs/phase56_corr/p6_clean_v2_kpos_macro_top1_over_uniform_vs_valid.svg

Quick Spearman vs valid MRR@20 (see p6_clean_v2_corr.csv):
- all: n_eff_expnorm +0.7017, cv +0.4662, top1_over_uniform -0.4804, knn_js +0.1491
- moe_top_k=0: n_eff_expnorm +0.7986, cv +0.5987, top1_over_uniform -0.7040, knn_js -0.0238
- moe_top_k>0: n_eff_expnorm +0.1610, cv -0.1610, top1_over_uniform +0.3506, knn_js +0.3197 (small n=14)