# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 11.3807 | 0.2333 | 0.2607 | 2.0863 | 0.0000 |
| micro@1 | 9.0095 | 0.5761 | 0.2878 | 1.4552 | 0.4568 |
| mid@1 | 9.5805 | 0.5025 | 0.3599 | 1.9707 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0241 | 0.9762 | 0.0123 | 0.9878 | 0.0123 | 0.9878 |
| micro@1 | 0.0350 | 0.9656 | 0.0153 | 0.9848 | 0.0204 | 0.9799 |
| mid@1 | 0.0188 | 0.9814 | 0.0064 | 0.9937 | 0.0103 | 0.9898 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0331 | 0.9675 | 0.0407 | 0.9601 | 0.0425 | 0.9584 | 0.0347 | 0.9659 | 0.1176 |
| micro@1 | 0.0591 | 0.9426 | 0.0687 | 0.9336 | 0.0699 | 0.9325 | 0.0655 | 0.9366 | 0.2006 |
| mid@1 | 0.0318 | 0.9687 | 0.0307 | 0.9698 | 0.0373 | 0.9634 | 0.0403 | 0.9605 | 0.1958 |
