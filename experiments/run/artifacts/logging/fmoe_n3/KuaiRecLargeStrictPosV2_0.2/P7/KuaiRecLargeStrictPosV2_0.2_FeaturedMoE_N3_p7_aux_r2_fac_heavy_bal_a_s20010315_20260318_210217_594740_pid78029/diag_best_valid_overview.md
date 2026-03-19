# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 11.2155 | 0.2645 | 0.1762 | 1.6899 | 0.0000 |
| micro@1 | 9.4919 | 0.5140 | 0.2397 | 1.4783 | 0.5973 |
| mid@1 | 8.6723 | 0.6194 | 0.1940 | 1.5461 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0480 | 0.9532 | 0.0435 | 0.9575 | 0.0057 | 0.9943 |
| micro@1 | 0.0479 | 0.9532 | 0.0428 | 0.9581 | 0.0067 | 0.9934 |
| mid@1 | 0.0332 | 0.9673 | 0.0265 | 0.9738 | 0.0076 | 0.9924 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0613 | 0.9406 | 0.0724 | 0.9301 | 0.0782 | 0.9248 | 0.0651 | 0.9370 | 0.1241 |
| micro@1 | 0.0764 | 0.9265 | 0.0912 | 0.9129 | 0.0939 | 0.9104 | 0.0808 | 0.9223 | 0.1625 |
| mid@1 | 0.0726 | 0.9300 | 0.0661 | 0.9360 | 0.0740 | 0.9286 | 0.0653 | 0.9368 | 0.1724 |
