# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 11.1438 | 0.2772 | 0.2900 | 1.9483 | 0.0000 |
| micro@1 | 11.6215 | 0.1805 | 0.1758 | 1.8389 | 0.5103 |
| mid@1 | 9.4702 | 0.5168 | 0.2429 | 1.7746 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0430 | 0.9579 | 0.0195 | 0.9806 | 0.0282 | 0.9722 |
| micro@1 | 0.0380 | 0.9628 | 0.0180 | 0.9822 | 0.0249 | 0.9754 |
| mid@1 | 0.0399 | 0.9609 | 0.0194 | 0.9808 | 0.0198 | 0.9804 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0539 | 0.9476 | 0.0582 | 0.9435 | 0.0632 | 0.9387 | 0.0559 | 0.9456 | 0.1395 |
| micro@1 | 0.0644 | 0.9376 | 0.0738 | 0.9289 | 0.0796 | 0.9235 | 0.0575 | 0.9441 | 0.1181 |
| mid@1 | 0.0601 | 0.9416 | 0.0581 | 0.9436 | 0.0699 | 0.9325 | 0.0674 | 0.9348 | 0.1770 |
