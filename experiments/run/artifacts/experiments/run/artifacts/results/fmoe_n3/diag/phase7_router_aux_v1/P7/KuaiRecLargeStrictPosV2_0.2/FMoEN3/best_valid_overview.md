# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 10.6881 | 0.3504 | 0.4021 | 1.8882 | 0.0000 |
| micro@1 | 11.1278 | 0.2800 | 0.3976 | 1.7367 | 0.3526 |
| mid@1 | 9.3942 | 0.5267 | 0.6908 | 2.1531 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0295 | 0.9709 | 0.0295 | 0.9709 | 0.0000 | 1.0000 |
| micro@1 | 0.0325 | 0.9680 | 0.0325 | 0.9680 | 0.0000 | 1.0000 |
| mid@1 | 0.0104 | 0.9897 | 0.0104 | 0.9897 | 0.0000 | 1.0000 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0382 | 0.9625 | 0.0434 | 0.9575 | 0.0466 | 0.9544 | 0.0374 | 0.9633 | 0.1161 |
| micro@1 | 0.0536 | 0.9478 | 0.0641 | 0.9379 | 0.0679 | 0.9344 | 0.0589 | 0.9428 | 0.1193 |
| mid@1 | 0.0158 | 0.9843 | 0.0142 | 0.9859 | 0.0148 | 0.9853 | 0.0229 | 0.9773 | 0.1633 |
