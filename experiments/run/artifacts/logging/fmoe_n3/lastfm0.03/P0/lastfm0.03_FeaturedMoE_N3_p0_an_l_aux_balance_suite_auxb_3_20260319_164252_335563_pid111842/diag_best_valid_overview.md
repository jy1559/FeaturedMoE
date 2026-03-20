# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 18.0458 | 0.3291 | 0.1206 | 1.4805 | 0.0000 |
| micro@1 | 19.5521 | 0.1514 | 0.0724 | 1.8511 | 0.5407 |
| mid@1 | 18.5566 | 0.2789 | 0.0821 | 1.3482 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.1206 | 0.8864 | 0.0561 | 0.9454 | 0.0835 | 0.9199 |
| micro@1 | 0.0789 | 0.9241 | 0.0246 | 0.9757 | 0.0584 | 0.9433 |
| mid@1 | 0.1086 | 0.8971 | 0.0537 | 0.9477 | 0.0715 | 0.9310 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.1784 | 0.8366 | 0.1759 | 0.8387 | 0.1785 | 0.8365 | 0.1645 | 0.8483 | 0.0969 |
| micro@1 | 0.2035 | 0.8159 | 0.2205 | 0.8022 | 0.2496 | 0.7791 | 0.1698 | 0.8438 | 0.0702 |
| mid@1 | 0.1771 | 0.8377 | 0.1815 | 0.8340 | 0.2127 | 0.8084 | 0.1646 | 0.8483 | 0.0762 |
