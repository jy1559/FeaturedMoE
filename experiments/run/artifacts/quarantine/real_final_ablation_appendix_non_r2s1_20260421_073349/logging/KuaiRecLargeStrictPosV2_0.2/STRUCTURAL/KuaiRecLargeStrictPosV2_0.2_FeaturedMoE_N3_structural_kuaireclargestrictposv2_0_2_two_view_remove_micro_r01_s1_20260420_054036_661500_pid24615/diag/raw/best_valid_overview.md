# Diagnostic Overview

- split: best_valid
- feature_mode: runtime:unknown|ablation:none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 16.1135 | 0.4911 | 0.2542 | 2.6541 | 0.8612 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0326 | 0.9679 | 0.0086 | 0.9915 | 0.0249 | 0.9754 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0400 | 0.9608 | 0.0400 | 0.9608 | 0.0408 | 0.9600 | 0.0378 | 0.9629 | 0.1154 |
