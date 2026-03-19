# P5+P6 Balance/Specialization Analysis (Input vs Output)

- source rows (P5+P6): 64
- filter applied: best_valid_mrr20 > 0.075
- filtered rows used: 63
- note: threshold 0.75 left too few rows, so 0.075 was used as practical fallback.

## Variable groups
- output metrics: n_eff_ratio_topkaware, cv_usage_mean, top1_over_uniform_mean, knn_score_mean, group_knn_score_mean, intra_group_knn_mean_score_mean
- input controls: balance_loss_lambda, z_loss_lambda, route_smoothness_lambda, route_sharpness_lambda, route_prior_lambda, route_monopoly_lambda, route_consistency_lambda, group_prior_align_lambda, factored_group_balance_lambda

## How top-k aware n_eff ratio was computed
- stage-wise denominator:
  - if topk_scope_mode starts with group: denominator = expert_scale
  - else if moe_top_k > 0: denominator = moe_top_k
  - else: denominator = expert_count (12)
- n_eff_ratio_topkaware = mean over stages of (stage_n_eff / denominator)
- top1_over_uniform_mean = mean over stages of (stage_top1_max_frac * denominator)

## Correlation summary (vs best_valid_mrr20)
- input group_prior_align_lambda: pearson=-0.5853, spearman=-0.5479, n=57
- input factored_group_balance_lambda: pearson=-0.4295, spearman=-0.3834, n=57
- input route_consistency_lambda: pearson=0.2374, spearman=0.2330, n=63
- input route_prior_lambda: pearson=0.1213, spearman=0.1503, n=63
- input route_sharpness_lambda: pearson=0.1860, spearman=0.1326, n=10
- input route_monopoly_lambda: pearson=0.1860, spearman=0.1326, n=10
- input balance_loss_lambda: pearson=-0.1257, spearman=-0.1162, n=63
- input z_loss_lambda: pearson=-0.1257, spearman=-0.1162, n=63
- input route_smoothness_lambda: pearson=0.0820, spearman=-0.0402, n=63
- output group_knn_score_mean: pearson=-0.5561, spearman=-0.6690, n=63
- output cv_usage_mean: pearson=0.3338, spearman=0.4667, n=63
- output knn_score_mean: pearson=-0.2750, spearman=-0.2504, n=63
- output n_eff_ratio_topkaware: pearson=0.3133, spearman=0.2206, n=63
- output top1_over_uniform_mean: pearson=0.3411, spearman=-0.0964, n=63
- output intra_group_knn_mean_score_mean: pearson=-0.1243, spearman=0.0449, n=63

## Input vs Output interpretation
- input variables are training knobs/regularization strengths; they influence routing behavior indirectly through optimization pressure.
- output variables are observed routing statistics measured after/within training (effective expert usage, concentration, neighborhood consistency).
- balance_loss_lambda typically pushes usage spread; z_loss_lambda stabilizes router logits; route_smoothness/sharpness/prior terms shape how peaked or smooth routing becomes.
- output metrics are computed from diag traces per stage and then aggregated (mean over macro/mid/micro in this report).

## Generated files
- filtered data: outputs/phase56_corr/p56_balance_specialization_filtered.csv
- correlations: outputs/phase56_corr/p56_balance_specialization_corr.csv
- scatter manifest: outputs/phase56_corr/p56_balance_specialization_plot_manifest.txt
