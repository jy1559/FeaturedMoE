# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 11.4427 | 0.2207 | 0.5205 | 2.3608 | 0.0000 |
| micro@1 | 5.5644 | 1.0754 | 0.8179 | 1.7343 | 0.2785 |
| mid@1 | 11.3902 | 0.2314 | 0.4702 | 2.3511 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0025 | 0.9975 | 0.0025 | 0.9975 | 0.0000 | 1.0000 |
| micro@1 | 0.0012 | 0.9988 | 0.0012 | 0.9988 | 0.0000 | 1.0000 |
| mid@1 | 0.0018 | 0.9982 | 0.0018 | 0.9982 | 0.0000 | 1.0000 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0056 | 0.9944 | 0.0097 | 0.9903 | 0.0105 | 0.9895 | 0.0068 | 0.9932 | 0.1083 |
| micro@1 | 0.0020 | 0.9980 | 0.0036 | 0.9964 | 0.0015 | 0.9985 | 0.0090 | 0.9910 | 0.2491 |
| mid@1 | 0.0072 | 0.9928 | 0.0048 | 0.9952 | 0.0062 | 0.9938 | 0.0110 | 0.9891 | 0.1056 |
