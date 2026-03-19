# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 10.8658 | 0.3231 | 0.2468 | 1.9607 | 0.0000 |
| micro@1 | 10.9745 | 0.3057 | 0.2705 | 1.9693 | 0.4625 |
| mid@1 | 10.1044 | 0.4331 | 0.2813 | 1.7738 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0395 | 0.9612 | 0.0172 | 0.9830 | 0.0267 | 0.9736 |
| micro@1 | 0.0305 | 0.9699 | 0.0118 | 0.9883 | 0.0203 | 0.9799 |
| mid@1 | 0.0337 | 0.9669 | 0.0156 | 0.9846 | 0.0180 | 0.9822 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0491 | 0.9520 | 0.0534 | 0.9480 | 0.0583 | 0.9434 | 0.0518 | 0.9495 | 0.1204 |
| micro@1 | 0.0551 | 0.9464 | 0.0672 | 0.9350 | 0.0736 | 0.9291 | 0.0485 | 0.9526 | 0.1507 |
| mid@1 | 0.0528 | 0.9486 | 0.0512 | 0.9501 | 0.0647 | 0.9373 | 0.0566 | 0.9450 | 0.1852 |
