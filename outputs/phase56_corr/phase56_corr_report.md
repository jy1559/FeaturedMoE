# Phase5/6 Diag-Performance Correlation (KuaiRecLargeStrictPosV2_0.2)

- total joined runs: 64
- P6 runs with diag: 63
- P5 runs with diag: 1
- primary target metric: best_valid_mrr20

## Top correlations vs best_valid_mrr20 (P6)

|metric|n|pearson_r|spearman_rho|
|---|---:|---:|---:|
|micro_1.n_eff|63|0.7665|0.7580|
|macro_1.n_eff|63|0.6978|0.7017|
|macro_1.entropy_mean|63|0.6662|0.6924|
|micro_1.entropy_mean|63|0.6159|0.6240|
|mid_1.family_top_expert_mean_share|63|0.4233|0.6222|
|mid_1.entropy_mean|63|0.5222|0.6076|
|mid_1.n_eff|63|0.5997|0.5816|
|micro_1.top1_max_frac|63|-0.5826|-0.5674|
|macro_1.family_top_expert_mean_share|63|0.3307|0.4982|
|micro_1.route_consistency_knn_js|63|0.3372|0.4947|
|micro_1.route_consistency_knn_score|63|-0.3405|-0.4947|
|macro_1.top1_max_frac|63|-0.4167|-0.4804|
|micro_1.family_top_expert_mean_share|63|0.3419|0.4736|
|macro_1.cv_usage|63|0.3396|0.4662|
|mid_1.top1_max_frac|63|-0.3242|-0.3950|

## Top correlations vs best_valid_test_mrr20 (P6)

|metric|n|pearson_r|spearman_rho|
|---|---:|---:|---:|
|mid_1.family_top_expert_mean_share|63|0.2347|0.5584|
|mid_1.cv_usage|63|0.1898|0.3669|
|macro_1.n_eff|63|-0.0692|0.3214|
|macro_1.entropy_mean|63|0.0418|0.3154|
|micro_1.n_eff|63|-0.0529|0.3020|
|micro_1.top1_max_frac|63|-0.0672|-0.2807|
|micro_1.entropy_mean|63|0.0133|0.2591|
|micro_1.route_jitter_adjacent|63|-0.3724|-0.1848|
|mid_1.route_consistency_knn_js|63|-0.1665|0.1712|
|mid_1.route_consistency_knn_score|63|0.1598|-0.1712|
|mid_1.top1_max_frac|63|0.3257|0.1485|
|micro_1.family_top_expert_mean_share|63|0.0027|0.1373|
|micro_1.route_consistency_knn_js|63|-0.2562|0.1222|
|micro_1.route_consistency_knn_score|63|0.2507|-0.1222|
|macro_1.top1_max_frac|63|0.1248|-0.1215|

## Notes
- P5 has only one run with diag_best_valid_overview.json, so P5 correlation is statistically invalid.
- Correlation is association, not causation.

## Interpretation for Specialization Claim
- P6(valid)에서는 `n_eff`(macro/micro)가 높을수록 MRR@20이 높게 나타남: `micro_1.n_eff` Spearman=0.758, `macro_1.n_eff` Spearman=0.702.
- 동시에 `top1_max_frac`(특정 expert 쏠림)는 높을수록 MRR@20이 낮아지는 경향: `micro_1.top1_max_frac` Spearman=-0.567, `macro_1.top1_max_frac` Spearman=-0.480.
- 즉, “무조건 균등”도 아니고 “과도한 쏠림”도 아닌, 다수 expert를 활용하는 구조적 specialization이 유리하다는 증거에 가깝다.
- KNN consistency는 stage별로 부호가 갈린다. 특히 `micro_1.route_consistency_knn_js`는 valid 기준 양(+) 상관(Spearman=0.495)이나 test 기준 약한 양/음 혼재가 있어, KNN 단일 지표만으로 일반화 성능을 단정하기 어렵다.

## Regularization Knob Coverage
- 이번 로깅 조인 테이블에서는 `z_loss_lambda`, `balance_loss_lambda`가 run-level로 거의 기록되지 않아 직접 상관 계산이 불가능했다.
- 따라서 “z-loss/load-balance 증가 시 성능 저하” 가설은 이번 결과로는 계량 검증이 부족하며, 해당 람다를 명시적으로 sweep한 추가 실험이 필요하다.

## Visualization Files
- `outputs/phase56_corr/p6_scatter_macro_knn_js_vs_valid.svg`
- `outputs/phase56_corr/p6_scatter_macro_knn_score_vs_valid.svg`
- `outputs/phase56_corr/p6_scatter_macro_n_eff_vs_valid.svg`
- `outputs/phase56_corr/p6_scatter_macro_cv_usage_vs_valid.svg`
- `outputs/phase56_corr/p6_scatter_macro_top1max_vs_valid.svg`
- `outputs/phase56_corr/p6_corr_top15_heatbars.svg`
- `outputs/phase56_corr/p6_family_summary.md`