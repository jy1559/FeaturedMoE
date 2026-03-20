# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 17.6177 | 0.3677 | 0.1321 | 1.4676 | 0.0000 |
| micro@1 | 19.5999 | 0.1429 | 0.0833 | 1.8583 | 0.5353 |
| mid@1 | 18.5286 | 0.2818 | 0.0885 | 1.3655 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.1197 | 0.8872 | 0.0572 | 0.9444 | 0.0817 | 0.9215 |
| micro@1 | 0.0790 | 0.9241 | 0.0252 | 0.9751 | 0.0581 | 0.9436 |
| mid@1 | 0.1132 | 0.8929 | 0.0563 | 0.9453 | 0.0735 | 0.9292 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.1782 | 0.8367 | 0.1756 | 0.8390 | 0.1785 | 0.8366 | 0.1640 | 0.8487 | 0.1056 |
| micro@1 | 0.2032 | 0.8161 | 0.2224 | 0.8006 | 0.2498 | 0.7790 | 0.1699 | 0.8438 | 0.0718 |
| mid@1 | 0.1794 | 0.8358 | 0.1883 | 0.8283 | 0.2149 | 0.8066 | 0.1662 | 0.8469 | 0.0797 |
