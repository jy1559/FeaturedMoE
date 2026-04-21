# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 15.7563 | 0.1244 | 0.1337 | 2.5542 | 0.7055 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0474 | 0.9537 | 0.0151 | 0.9850 | 0.0327 | 0.9678 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0582 | 0.9435 | 0.0575 | 0.9441 | 0.0570 | 0.9446 | 0.0527 | 0.9486 | 0.0836 |
