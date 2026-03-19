# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 5.9212 | 0.1154 | 0.2751 | 1.4536 | 0.0000 |
| micro@1 | 5.1941 | 0.3939 | 0.3382 | 1.1498 | 0.4194 |
| mid@1 | 4.6074 | 0.5498 | 0.5254 | 1.0225 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0244 | 0.9759 | 0.0068 | 0.9932 | 0.0189 | 0.9813 |
| micro@1 | 0.0252 | 0.9751 | 0.0066 | 0.9934 | 0.0194 | 0.9808 |
| mid@1 | 0.0273 | 0.9731 | 0.0046 | 0.9954 | 0.0197 | 0.9805 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0284 | 0.9720 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0955 |
| micro@1 | 0.0395 | 0.9613 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1343 |
| mid@1 | 0.0477 | 0.9534 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1832 |
