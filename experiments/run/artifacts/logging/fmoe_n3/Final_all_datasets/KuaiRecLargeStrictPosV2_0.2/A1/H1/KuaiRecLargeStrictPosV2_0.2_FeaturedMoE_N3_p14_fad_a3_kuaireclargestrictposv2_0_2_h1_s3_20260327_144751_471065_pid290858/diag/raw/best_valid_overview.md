# Diagnostic Overview

- split: best_valid
- feature_mode: perturb:zero@both:kw=cat+theme#shift1

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 11.2386 | 0.2603 | 0.3528 | 2.3372 | 0.6174 |
| micro@1 | 8.4779 | 0.6445 | 0.4928 | 2.0187 | 0.5370 |
| mid@1 | 9.1167 | 0.5624 | 0.5007 | 2.0069 | 0.5083 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0125 | 0.9876 | 0.0070 | 0.9930 | 0.0053 | 0.9947 |
| micro@1 | 0.0205 | 0.9797 | 0.0187 | 0.9815 | 0.0034 | 0.9966 |
| mid@1 | 0.0227 | 0.9776 | 0.0190 | 0.9812 | 0.0041 | 0.9959 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0214 | 0.9789 | 0.0238 | 0.9765 | 0.0151 | 0.9850 | 0.0252 | 0.9751 | 0.0905 |
| micro@1 | 0.0431 | 0.9578 | 0.0383 | 0.9624 | 0.0418 | 0.9590 | 0.0202 | 0.9800 | 0.1294 |
| mid@1 | 0.0659 | 0.9363 | 0.0515 | 0.9498 | 0.0758 | 0.9270 | 0.0304 | 0.9701 | 0.1557 |
