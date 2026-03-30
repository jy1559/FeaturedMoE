# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 9.9577 | 0.4529 | 0.2680 | 2.1719 | 0.5073 |
| micro@1 | 6.1025 | 0.9831 | 0.4990 | 1.7169 | 0.4644 |
| mid@1 | 8.5473 | 0.6356 | 0.3245 | 1.9894 | 0.4932 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0259 | 0.9744 | 0.0046 | 0.9954 | 0.0193 | 0.9809 |
| micro@1 | 0.0132 | 0.9868 | 0.0110 | 0.9890 | 0.0059 | 0.9941 |
| mid@1 | 0.0286 | 0.9718 | 0.0147 | 0.9854 | 0.0152 | 0.9849 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0370 | 0.9637 | 0.0598 | 0.9420 | 0.0449 | 0.9561 | 0.0554 | 0.9461 | 0.1446 |
| micro@1 | 0.0252 | 0.9751 | 0.0250 | 0.9753 | 0.0248 | 0.9755 | 0.0159 | 0.9842 | 0.2414 |
| mid@1 | 0.0709 | 0.9316 | 0.0663 | 0.9359 | 0.0870 | 0.9167 | 0.0600 | 0.9418 | 0.2065 |
