# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 5.8504 | 0.1599 | 0.2512 | 1.4897 | 0.0000 |
| micro@1 | 5.6059 | 0.2651 | 0.3003 | 1.3294 | 0.4802 |
| mid@1 | 4.7615 | 0.5100 | 0.4866 | 1.1435 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0281 | 0.9723 | 0.0073 | 0.9927 | 0.0209 | 0.9793 |
| micro@1 | 0.0231 | 0.9772 | 0.0055 | 0.9946 | 0.0177 | 0.9825 |
| mid@1 | 0.0222 | 0.9781 | 0.0042 | 0.9959 | 0.0138 | 0.9863 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0000 | 1.0000 | 0.0354 | 0.9652 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0990 |
| micro@1 | 0.0000 | 1.0000 | 0.0421 | 0.9588 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1194 |
| mid@1 | 0.0000 | 1.0000 | 0.0328 | 0.9678 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1712 |
