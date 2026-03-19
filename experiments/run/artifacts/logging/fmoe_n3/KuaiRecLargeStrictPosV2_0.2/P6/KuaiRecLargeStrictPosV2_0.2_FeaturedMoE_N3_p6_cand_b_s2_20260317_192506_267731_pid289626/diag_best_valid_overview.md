# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 11.0613 | 0.2913 | 0.2788 | 1.9718 | 0.0000 |
| micro@1 | 9.6823 | 0.4893 | 0.3046 | 1.6588 | 0.4911 |
| mid@1 | 9.9338 | 0.4561 | 0.3149 | 1.9457 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0262 | 0.9742 | 0.0127 | 0.9874 | 0.0132 | 0.9869 |
| micro@1 | 0.0289 | 0.9715 | 0.0120 | 0.9881 | 0.0177 | 0.9825 |
| mid@1 | 0.0209 | 0.9793 | 0.0072 | 0.9929 | 0.0116 | 0.9885 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0363 | 0.9643 | 0.0444 | 0.9566 | 0.0466 | 0.9545 | 0.0375 | 0.9632 | 0.1309 |
| micro@1 | 0.0472 | 0.9539 | 0.0551 | 0.9464 | 0.0539 | 0.9476 | 0.0577 | 0.9439 | 0.1832 |
| mid@1 | 0.0349 | 0.9657 | 0.0342 | 0.9664 | 0.0431 | 0.9578 | 0.0427 | 0.9582 | 0.1798 |
