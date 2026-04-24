# Diagnostic Overview

- split: best_valid
- feature_mode: runtime:unknown|ablation:none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 15.7481 | 0.1265 | 0.1096 | 2.5385 | 0.7367 |
| mid@1 | 15.7772 | 0.1188 | 0.1169 | 2.4703 | 0.6865 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0584 | 0.9433 | 0.0146 | 0.9855 | 0.0440 | 0.9569 |
| mid@1 | 0.0647 | 0.9373 | 0.0151 | 0.9850 | 0.0496 | 0.9516 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0683 | 0.9340 | 0.0674 | 0.9348 | 0.0662 | 0.9360 | 0.0636 | 0.9384 | 0.0812 |
| mid@1 | 0.0824 | 0.9209 | 0.0817 | 0.9215 | 0.0773 | 0.9256 | 0.0759 | 0.9269 | 0.0824 |
