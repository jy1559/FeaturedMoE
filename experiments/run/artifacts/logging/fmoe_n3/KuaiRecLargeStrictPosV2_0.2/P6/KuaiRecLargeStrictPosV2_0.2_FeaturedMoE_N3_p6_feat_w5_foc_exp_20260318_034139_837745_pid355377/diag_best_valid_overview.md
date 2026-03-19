# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 5.8852 | 0.1397 | 0.3061 | 1.3502 | 0.0000 |
| micro@1 | 5.5654 | 0.2795 | 0.3272 | 1.1613 | 0.5277 |
| mid@1 | 5.6375 | 0.2536 | 0.3701 | 1.4377 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0328 | 0.9678 | 0.0085 | 0.9916 | 0.0249 | 0.9754 |
| micro@1 | 0.0309 | 0.9696 | 0.0083 | 0.9917 | 0.0235 | 0.9767 |
| mid@1 | 0.0170 | 0.9831 | 0.0078 | 0.9922 | 0.0092 | 0.9909 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0000 | 1.0000 | 0.0482 | 0.9529 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0983 |
| micro@1 | 0.0000 | 1.0000 | 0.0576 | 0.9440 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1481 |
| mid@1 | 0.0000 | 1.0000 | 0.0209 | 0.9794 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1292 |
