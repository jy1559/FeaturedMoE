# Diagnostic Overview

- split: best_valid
- feature_mode: runtime:unknown|ablation:none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 15.7556 | 0.1246 | 0.1264 | 2.6026 | 0.7259 |
| mid@1 | 15.8673 | 0.0914 | 0.1043 | 2.5988 | 0.7227 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0396 | 0.9611 | 0.0081 | 0.9920 | 0.0314 | 0.9691 |
| mid@1 | 0.0401 | 0.9607 | 0.0096 | 0.9904 | 0.0307 | 0.9698 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0487 | 0.9525 | 0.0482 | 0.9529 | 0.0471 | 0.9540 | 0.0432 | 0.9577 | 0.0779 |
| mid@1 | 0.0488 | 0.9524 | 0.0475 | 0.9536 | 0.0492 | 0.9520 | 0.0452 | 0.9558 | 0.0757 |
