# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 11.3979 | 0.2298 | 0.2421 | 2.3108 | 0.0000 |
| micro@1 | 10.3454 | 0.3999 | 0.5901 | 2.0410 | 0.4133 |
| mid@1 | 11.8452 | 0.1143 | 0.2900 | 2.4274 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0128 | 0.9873 | 0.0066 | 0.9934 | 0.0067 | 0.9933 |
| micro@1 | 0.0052 | 0.9948 | 0.0035 | 0.9965 | 0.0041 | 0.9959 |
| mid@1 | 0.0018 | 0.9982 | 0.0015 | 0.9985 | 0.0003 | 0.9997 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0164 | 0.9838 | 0.0224 | 0.9779 | 0.0204 | 0.9798 | 0.0226 | 0.9776 | 0.1216 |
| micro@1 | 0.0061 | 0.9939 | 0.0115 | 0.9886 | 0.0056 | 0.9944 | 0.0185 | 0.9817 | 0.1383 |
| mid@1 | 0.0038 | 0.9962 | 0.0037 | 0.9963 | 0.0042 | 0.9958 | 0.0075 | 0.9925 | 0.0952 |
