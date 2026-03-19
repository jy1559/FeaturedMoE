# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 11.4654 | 0.2159 | 0.1486 | 1.6337 | 0.0000 |
| micro@1 | 10.3101 | 0.4048 | 0.1611 | 1.4335 | 0.6037 |
| mid@1 | 9.5396 | 0.5078 | 0.1993 | 1.6325 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0502 | 0.9510 | 0.0452 | 0.9558 | 0.0062 | 0.9938 |
| micro@1 | 0.0485 | 0.9527 | 0.0426 | 0.9583 | 0.0097 | 0.9904 |
| mid@1 | 0.0369 | 0.9638 | 0.0287 | 0.9717 | 0.0098 | 0.9903 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0641 | 0.9379 | 0.0775 | 0.9255 | 0.0833 | 0.9201 | 0.0697 | 0.9326 | 0.1123 |
| micro@1 | 0.0779 | 0.9250 | 0.0944 | 0.9099 | 0.0986 | 0.9061 | 0.0828 | 0.9205 | 0.1415 |
| mid@1 | 0.0797 | 0.9234 | 0.0743 | 0.9283 | 0.0834 | 0.9200 | 0.0682 | 0.9341 | 0.1554 |
