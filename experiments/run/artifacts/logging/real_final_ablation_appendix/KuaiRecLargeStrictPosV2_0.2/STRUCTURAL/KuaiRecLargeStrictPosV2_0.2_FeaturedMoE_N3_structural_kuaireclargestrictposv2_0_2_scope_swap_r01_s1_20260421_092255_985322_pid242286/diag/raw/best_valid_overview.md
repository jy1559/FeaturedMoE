# Diagnostic Overview

- split: best_valid
- feature_mode: runtime:unknown|ablation:none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 15.2405 | 0.5588 | 0.1711 | 2.2380 | 0.8983 |
| mid@1 | 8.5471 | 1.1576 | 0.3877 | 1.8366 | 0.8062 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0625 | 0.9394 | 0.0151 | 0.9850 | 0.0482 | 0.9529 |
| mid@1 | 0.0663 | 0.9358 | 0.0179 | 0.9822 | 0.0524 | 0.9490 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0792 | 0.9238 | 0.0814 | 0.9218 | 0.0825 | 0.9208 | 0.0755 | 0.9273 | 0.1148 |
| mid@1 | 0.0927 | 0.9115 | 0.0916 | 0.9125 | 0.0867 | 0.9170 | 0.0764 | 0.9264 | 0.2684 |
