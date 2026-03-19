# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 11.4231 | 0.2247 | 0.5161 | 2.3716 | 0.0000 |
| micro@1 | 5.7827 | 1.0369 | 0.8179 | 1.8163 | 0.2785 |
| mid@1 | 11.3515 | 0.2390 | 0.4451 | 2.3615 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0022 | 0.9978 | 0.0022 | 0.9978 | 0.0000 | 1.0000 |
| micro@1 | 0.0010 | 0.9990 | 0.0010 | 0.9990 | 0.0000 | 1.0000 |
| mid@1 | 0.0016 | 0.9984 | 0.0016 | 0.9984 | 0.0000 | 1.0000 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0050 | 0.9950 | 0.0084 | 0.9916 | 0.0091 | 0.9909 | 0.0058 | 0.9942 | 0.1071 |
| micro@1 | 0.0018 | 0.9982 | 0.0027 | 0.9973 | 0.0014 | 0.9986 | 0.0076 | 0.9924 | 0.2429 |
| mid@1 | 0.0065 | 0.9935 | 0.0043 | 0.9958 | 0.0056 | 0.9944 | 0.0096 | 0.9904 | 0.1068 |
