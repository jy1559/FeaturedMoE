# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 2.9565 | 0.1213 | 0.3978 | 0.8999 | 0.0000 |
| micro@1 | 2.9585 | 0.1184 | 0.3778 | 0.8521 | 0.5052 |
| mid@1 | 2.4987 | 0.4479 | 0.6115 | 0.3505 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0200 | 0.9802 | 0.0000 | 1.0000 | 0.0200 | 0.9802 |
| micro@1 | 0.0148 | 0.9853 | -0.0000 | 1.0000 | 0.0148 | 0.9853 |
| mid@1 | 0.0214 | 0.9788 | 0.0000 | 1.0000 | 0.0214 | 0.9788 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0892 |
| micro@1 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0961 |
| mid@1 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1425 |
