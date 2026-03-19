# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 9.9789 | 0.4500 | 0.1535 | 1.5244 | 0.0000 |
| micro@1 | 10.4082 | 0.3911 | 0.1841 | 1.5677 | 0.4718 |
| mid@1 | 10.6289 | 0.3592 | 0.1695 | 1.4406 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0748 | 0.9279 | 0.0315 | 0.9690 | 0.0674 | 0.9349 |
| micro@1 | 0.0453 | 0.9557 | 0.0152 | 0.9849 | 0.0423 | 0.9586 |
| mid@1 | 0.0696 | 0.9328 | 0.0299 | 0.9705 | 0.0691 | 0.9333 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.1120 | 0.8941 | 0.1059 | 0.8995 | 0.1100 | 0.8958 | 0.1025 | 0.9026 | 0.1402 |
| micro@1 | 0.1427 | 0.8670 | 0.1444 | 0.8655 | 0.1881 | 0.8285 | 0.1041 | 0.9012 | 0.1394 |
| mid@1 | 0.1300 | 0.8781 | 0.1053 | 0.9001 | 0.1463 | 0.8639 | 0.1223 | 0.8849 | 0.1411 |
