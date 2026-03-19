# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 3.4663 | 1.5690 | 0.5176 | 0.6456 | 0.0000 |
| micro@1 | 4.1555 | 1.3739 | 0.4005 | 0.4659 | 0.2954 |
| mid@1 | 1.7329 | 2.4341 | 0.7577 | 0.3532 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0716 | 0.9309 | 0.0348 | 0.9658 | 0.0297 | 0.9708 |
| micro@1 | 0.0471 | 0.9540 | 0.0246 | 0.9757 | 0.0413 | 0.9596 |
| mid@1 | 0.0121 | 0.9880 | 0.0081 | 0.9920 | 0.0145 | 0.9857 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0908 | 0.9132 | 0.1000 | 0.9049 | 0.1102 | 0.8956 | 0.0956 | 0.9089 | 0.4572 |
| micro@1 | 0.0812 | 0.9220 | 0.0936 | 0.9106 | 0.0959 | 0.9086 | 0.0741 | 0.9286 | 0.4034 |
| mid@1 | 0.0283 | 0.9721 | 0.0244 | 0.9759 | 0.0214 | 0.9788 | 0.0654 | 0.9367 | 0.7741 |
