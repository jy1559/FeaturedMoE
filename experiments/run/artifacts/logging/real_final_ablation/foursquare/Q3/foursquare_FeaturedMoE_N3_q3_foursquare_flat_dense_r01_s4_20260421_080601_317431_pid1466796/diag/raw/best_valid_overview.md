# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 15.7822 | 0.1175 | 0.0959 | 2.4556 | 0.7206 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0779 | 0.9251 | 0.0218 | 0.9785 | 0.0563 | 0.9453 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0921 | 0.9120 | 0.0894 | 0.9145 | 0.0894 | 0.9144 | 0.0834 | 0.9200 | 0.0785 |
