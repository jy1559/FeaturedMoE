# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 2.9707 | 0.0993 | 0.4240 | 0.8418 | 0.0000 |
| micro@1 | 2.7980 | 0.2687 | 0.5433 | 0.7950 | 0.4243 |
| mid@1 | 2.4670 | 0.4648 | 0.6300 | 0.4723 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0251 | 0.9752 | 0.0000 | 1.0000 | 0.0251 | 0.9752 |
| micro@1 | 0.0178 | 0.9824 | -0.0000 | 1.0000 | 0.0178 | 0.9824 |
| mid@1 | 0.0238 | 0.9765 | 0.0000 | 1.0000 | 0.0238 | 0.9765 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0911 |
| micro@1 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1249 |
| mid@1 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1433 |
