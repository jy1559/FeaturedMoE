# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 11.6645 | 0.1696 | 0.1779 | 2.1031 | 0.0000 |
| micro@1 | 9.1952 | 0.5523 | 0.2639 | 1.7947 | 0.6435 |
| mid@1 | 9.5689 | 0.5040 | 0.2764 | 1.8963 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0112 | 0.9889 | 0.0063 | 0.9937 | 0.0046 | 0.9954 |
| micro@1 | 0.0064 | 0.9936 | 0.0040 | 0.9960 | 0.0030 | 0.9970 |
| mid@1 | 0.0140 | 0.9861 | 0.0080 | 0.9921 | 0.0043 | 0.9957 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0245 | 0.9758 | 0.0400 | 0.9608 | 0.0450 | 0.9560 | 0.0286 | 0.9718 | 0.1100 |
| micro@1 | 0.0148 | 0.9853 | 0.0217 | 0.9786 | 0.0216 | 0.9786 | 0.0180 | 0.9822 | 0.1726 |
| mid@1 | 0.0436 | 0.9573 | 0.0330 | 0.9675 | 0.0435 | 0.9574 | 0.0499 | 0.9513 | 0.1656 |
