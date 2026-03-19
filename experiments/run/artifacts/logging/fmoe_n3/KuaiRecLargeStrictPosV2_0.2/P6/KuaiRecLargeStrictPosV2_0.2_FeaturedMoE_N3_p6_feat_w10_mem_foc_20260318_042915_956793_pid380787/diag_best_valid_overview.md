# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 5.8214 | 0.1751 | 0.3262 | 1.4439 | 0.0000 |
| micro@1 | 5.3835 | 0.3384 | 0.3839 | 1.2334 | 0.4510 |
| mid@1 | 4.9090 | 0.4714 | 0.4092 | 1.1280 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0252 | 0.9751 | 0.0051 | 0.9949 | 0.0202 | 0.9800 |
| micro@1 | 0.0217 | 0.9785 | 0.0052 | 0.9948 | 0.0168 | 0.9834 |
| mid@1 | 0.0269 | 0.9735 | 0.0058 | 0.9943 | 0.0156 | 0.9846 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0000 | 1.0000 | 0.0317 | 0.9688 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0989 |
| micro@1 | 0.0000 | 1.0000 | 0.0381 | 0.9626 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1370 |
| mid@1 | 0.0000 | 1.0000 | 0.0408 | 0.9600 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1592 |
