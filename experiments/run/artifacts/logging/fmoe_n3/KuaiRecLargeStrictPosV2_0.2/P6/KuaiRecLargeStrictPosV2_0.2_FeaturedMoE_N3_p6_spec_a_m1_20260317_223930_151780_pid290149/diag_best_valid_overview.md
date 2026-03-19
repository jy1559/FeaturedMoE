# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 10.2729 | 0.4100 | 0.2373 | 1.7519 | 0.0000 |
| micro@1 | 11.3127 | 0.2465 | 0.1602 | 1.9260 | 0.4422 |
| mid@1 | 9.0976 | 0.5648 | 0.3181 | 1.4707 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0530 | 0.9484 | 0.0259 | 0.9744 | 0.0343 | 0.9663 |
| micro@1 | 0.0349 | 0.9657 | 0.0165 | 0.9836 | 0.0205 | 0.9797 |
| mid@1 | 0.0425 | 0.9584 | 0.0201 | 0.9801 | 0.0223 | 0.9780 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0656 | 0.9366 | 0.0721 | 0.9305 | 0.0789 | 0.9241 | 0.0697 | 0.9326 | 0.1500 |
| micro@1 | 0.0632 | 0.9388 | 0.0778 | 0.9252 | 0.0844 | 0.9191 | 0.0582 | 0.9435 | 0.1175 |
| mid@1 | 0.0671 | 0.9351 | 0.0661 | 0.9360 | 0.0787 | 0.9243 | 0.0742 | 0.9285 | 0.2148 |
