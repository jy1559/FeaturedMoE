# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 11.2443 | 0.2592 | 0.5673 | 2.3589 | 0.0000 |
| micro@1 | 6.1234 | 0.9796 | 0.8178 | 1.8691 | 0.2785 |
| mid@1 | 11.2437 | 0.2594 | 0.4508 | 2.3429 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0023 | 0.9977 | 0.0023 | 0.9977 | 0.0000 | 1.0000 |
| micro@1 | 0.0009 | 0.9991 | 0.0009 | 0.9991 | 0.0000 | 1.0000 |
| mid@1 | 0.0017 | 0.9983 | 0.0017 | 0.9983 | 0.0000 | 1.0000 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0054 | 0.9946 | 0.0086 | 0.9914 | 0.0094 | 0.9906 | 0.0057 | 0.9943 | 0.1139 |
| micro@1 | 0.0017 | 0.9983 | 0.0027 | 0.9973 | 0.0015 | 0.9986 | 0.0075 | 0.9926 | 0.2343 |
| mid@1 | 0.0074 | 0.9926 | 0.0049 | 0.9951 | 0.0062 | 0.9938 | 0.0109 | 0.9891 | 0.1084 |
