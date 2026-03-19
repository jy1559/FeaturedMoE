# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 11.5153 | 0.2052 | 0.1742 | 2.0369 | 0.0000 |
| micro@1 | 8.9854 | 0.5792 | 0.2541 | 1.7342 | 0.6421 |
| mid@1 | 9.6477 | 0.4938 | 0.2796 | 1.8774 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0132 | 0.9869 | 0.0070 | 0.9930 | 0.0059 | 0.9942 |
| micro@1 | 0.0073 | 0.9928 | 0.0043 | 0.9957 | 0.0037 | 0.9963 |
| mid@1 | 0.0148 | 0.9853 | 0.0086 | 0.9915 | 0.0045 | 0.9955 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0283 | 0.9721 | 0.0473 | 0.9538 | 0.0525 | 0.9489 | 0.0341 | 0.9665 | 0.1169 |
| micro@1 | 0.0167 | 0.9834 | 0.0244 | 0.9759 | 0.0245 | 0.9758 | 0.0207 | 0.9795 | 0.1742 |
| mid@1 | 0.0461 | 0.9550 | 0.0347 | 0.9659 | 0.0464 | 0.9547 | 0.0535 | 0.9479 | 0.1662 |
