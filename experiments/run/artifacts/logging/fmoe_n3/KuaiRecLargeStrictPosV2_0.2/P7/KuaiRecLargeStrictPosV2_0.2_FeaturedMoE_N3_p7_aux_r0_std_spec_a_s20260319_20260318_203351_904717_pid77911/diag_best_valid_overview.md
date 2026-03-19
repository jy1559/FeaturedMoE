# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 11.3486 | 0.2396 | 0.2161 | 1.9155 | 0.0000 |
| micro@1 | 11.2549 | 0.2573 | 0.1848 | 1.9665 | 0.4950 |
| mid@1 | 8.7629 | 0.6078 | 0.3566 | 1.5918 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0468 | 0.9543 | 0.0215 | 0.9787 | 0.0299 | 0.9705 |
| micro@1 | 0.0309 | 0.9695 | 0.0123 | 0.9878 | 0.0203 | 0.9800 |
| mid@1 | 0.0436 | 0.9574 | 0.0244 | 0.9759 | 0.0213 | 0.9789 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0575 | 0.9441 | 0.0628 | 0.9391 | 0.0680 | 0.9342 | 0.0602 | 0.9416 | 0.1260 |
| micro@1 | 0.0551 | 0.9464 | 0.0655 | 0.9366 | 0.0710 | 0.9315 | 0.0483 | 0.9528 | 0.1177 |
| mid@1 | 0.0661 | 0.9360 | 0.0633 | 0.9387 | 0.0755 | 0.9273 | 0.0729 | 0.9297 | 0.2272 |
