# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 6.0008 | 0.9999 | 0.4230 | 1.8398 | 0.3219 |
| micro@1 | 2.6606 | 1.8736 | 0.9998 | 1.0722 | 0.0001 |
| mid@1 | 8.1015 | 0.6937 | 0.8483 | 2.1800 | 0.1787 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0133 | 0.9868 | 0.0103 | 0.9897 | 0.0030 | 0.9970 |
| micro@1 | 0.0021 | 0.9979 | 0.0020 | 0.9980 | 0.0037 | 0.9964 |
| mid@1 | 0.0088 | 0.9912 | 0.0028 | 0.9972 | 0.0069 | 0.9931 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0212 | 0.9790 | 0.0302 | 0.9702 | 0.0303 | 0.9702 | 0.0267 | 0.9737 | 0.2529 |
| micro@1 | 0.0052 | 0.9948 | 0.0063 | 0.9937 | 0.0050 | 0.9950 | 0.0027 | 0.9973 | 0.5127 |
| mid@1 | 0.0270 | 0.9733 | 0.0257 | 0.9746 | 0.0252 | 0.9751 | 0.0195 | 0.9807 | 0.2505 |
