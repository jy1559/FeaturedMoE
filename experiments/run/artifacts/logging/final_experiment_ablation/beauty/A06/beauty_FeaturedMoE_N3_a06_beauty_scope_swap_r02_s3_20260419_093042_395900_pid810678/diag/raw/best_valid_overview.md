# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 15.4952 | 0.1805 | 0.1852 | 2.4489 | 0.7747 |
| mid@1 | 14.4376 | 0.3290 | 0.1594 | 2.3555 | 0.7559 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0482 | 0.9529 | 0.0119 | 0.9881 | 0.0368 | 0.9639 |
| mid@1 | 0.0441 | 0.9569 | 0.0101 | 0.9900 | 0.0352 | 0.9654 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0574 | 0.9442 | 0.0636 | 0.9384 | 0.0641 | 0.9379 | 0.0617 | 0.9402 | 0.0886 |
| mid@1 | 0.0613 | 0.9405 | 0.0800 | 0.9231 | 0.0787 | 0.9244 | 0.0515 | 0.9498 | 0.1033 |
