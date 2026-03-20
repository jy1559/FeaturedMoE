# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 4.5284 | 1.8484 | 0.3322 | 0.9044 | 0.0000 |
| micro@1 | 15.3947 | 0.5469 | 0.1150 | 1.1459 | 0.4431 |
| mid@1 | 5.2636 | 1.6732 | 0.2706 | 0.8544 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0586 | 0.9431 | 0.0333 | 0.9673 | 0.0101 | 0.9901 |
| micro@1 | 0.1078 | 0.8978 | 0.0441 | 0.9569 | 0.0787 | 0.9243 |
| mid@1 | 0.0769 | 0.9260 | 0.0552 | 0.9463 | 0.0265 | 0.9742 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.1084 | 0.8973 | 0.1050 | 0.9003 | 0.1071 | 0.8984 | 0.0933 | 0.9109 | 0.3143 |
| micro@1 | 0.2897 | 0.7485 | 0.3219 | 0.7248 | 0.3555 | 0.7008 | 0.2279 | 0.7962 | 0.1313 |
| mid@1 | 0.1417 | 0.8679 | 0.1365 | 0.8724 | 0.1682 | 0.8452 | 0.1233 | 0.8840 | 0.2580 |
