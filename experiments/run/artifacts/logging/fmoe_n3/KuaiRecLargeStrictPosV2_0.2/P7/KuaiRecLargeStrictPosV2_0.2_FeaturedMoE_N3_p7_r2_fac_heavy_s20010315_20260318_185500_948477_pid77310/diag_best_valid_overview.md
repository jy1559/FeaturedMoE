# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 11.2670 | 0.2551 | 0.1684 | 1.6540 | 0.0000 |
| micro@1 | 9.9280 | 0.4568 | 0.2471 | 1.4607 | 0.5731 |
| mid@1 | 8.4978 | 0.6420 | 0.1978 | 1.4965 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0497 | 0.9515 | 0.0449 | 0.9561 | 0.0061 | 0.9940 |
| micro@1 | 0.0481 | 0.9531 | 0.0425 | 0.9584 | 0.0089 | 0.9912 |
| mid@1 | 0.0339 | 0.9667 | 0.0267 | 0.9737 | 0.0085 | 0.9915 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0635 | 0.9385 | 0.0760 | 0.9269 | 0.0818 | 0.9215 | 0.0682 | 0.9341 | 0.1206 |
| micro@1 | 0.0784 | 0.9246 | 0.0948 | 0.9096 | 0.0968 | 0.9078 | 0.0828 | 0.9205 | 0.1519 |
| mid@1 | 0.0766 | 0.9263 | 0.0701 | 0.9323 | 0.0780 | 0.9250 | 0.0672 | 0.9350 | 0.1743 |
