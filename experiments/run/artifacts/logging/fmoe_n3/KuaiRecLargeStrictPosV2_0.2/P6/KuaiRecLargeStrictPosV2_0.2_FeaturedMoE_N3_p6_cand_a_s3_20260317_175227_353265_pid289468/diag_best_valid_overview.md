# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 10.8807 | 0.3207 | 0.2400 | 1.8180 | 0.0000 |
| micro@1 | 11.3216 | 0.2448 | 0.1528 | 1.7293 | 0.5155 |
| mid@1 | 9.2584 | 0.5442 | 0.2845 | 1.4571 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0464 | 0.9546 | 0.0194 | 0.9808 | 0.0312 | 0.9693 |
| micro@1 | 0.0416 | 0.9592 | 0.0191 | 0.9811 | 0.0252 | 0.9751 |
| mid@1 | 0.0360 | 0.9647 | 0.0189 | 0.9813 | 0.0208 | 0.9794 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0575 | 0.9441 | 0.0643 | 0.9377 | 0.0705 | 0.9320 | 0.0622 | 0.9397 | 0.1345 |
| micro@1 | 0.0761 | 0.9268 | 0.0881 | 0.9157 | 0.0992 | 0.9056 | 0.0631 | 0.9388 | 0.1229 |
| mid@1 | 0.0576 | 0.9440 | 0.0544 | 0.9471 | 0.0641 | 0.9379 | 0.0694 | 0.9330 | 0.1819 |
