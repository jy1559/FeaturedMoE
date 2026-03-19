# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 10.2599 | 0.4118 | 0.1502 | 1.5562 | 0.0000 |
| micro@1 | 9.0970 | 0.5649 | 0.2350 | 1.3392 | 0.4027 |
| mid@1 | 7.6996 | 0.7473 | 0.3520 | 1.3392 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0692 | 0.9332 | 0.0280 | 0.9724 | 0.0730 | 0.9297 |
| micro@1 | 0.0496 | 0.9516 | 0.0237 | 0.9765 | 0.0551 | 0.9466 |
| mid@1 | 0.0489 | 0.9523 | 0.0244 | 0.9759 | 0.0775 | 0.9256 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0830 | 0.9204 | 0.0901 | 0.9138 | 0.0975 | 0.9071 | 0.0841 | 0.9193 | 0.1391 |
| micro@1 | 0.0849 | 0.9186 | 0.0992 | 0.9056 | 0.1115 | 0.8945 | 0.0712 | 0.9313 | 0.1806 |
| mid@1 | 0.0724 | 0.9301 | 0.0665 | 0.9357 | 0.0782 | 0.9248 | 0.0953 | 0.9091 | 0.2351 |
