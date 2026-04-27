# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| mid@1 | 15.4239 | 0.1933 | 0.1036 | 1.8812 | 0.6595 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| mid@1 | 0.0956 | 0.9088 | 0.0287 | 0.9717 | 0.0702 | 0.9322 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mid@1 | 0.1492 | 0.8614 | 0.1590 | 0.8530 | 0.1761 | 0.8385 | 0.1154 | 0.8910 | 0.0917 |
