# Diagnostic Overview

- split: best_valid
- feature_mode: runtime:unknown|ablation:none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 3.9970 | 0.0274 | 0.3222 | 1.3449 | 0.6177 |
| mid@1 | 3.9895 | 0.0513 | 0.2930 | 1.3343 | 0.5728 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0053 | 0.9947 | 0.0053 | 0.9947 | 0.0000 | 1.0000 |
| mid@1 | 0.0069 | 0.9931 | 0.0069 | 0.9931 | 0.0000 | 1.0000 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0061 | 0.9939 | 0.0059 | 0.9941 | 0.0061 | 0.9939 | 0.0061 | 0.9939 | 0.2586 |
| mid@1 | 0.0100 | 0.9900 | 0.0102 | 0.9898 | 0.0097 | 0.9903 | 0.0082 | 0.9918 | 0.2601 |
