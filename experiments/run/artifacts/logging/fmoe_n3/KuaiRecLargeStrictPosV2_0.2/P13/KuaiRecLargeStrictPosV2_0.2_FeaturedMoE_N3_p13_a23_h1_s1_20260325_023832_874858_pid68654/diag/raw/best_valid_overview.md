# Diagnostic Overview

- split: best_valid
- feature_mode: perturb:position_shift@train#shift3

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 11.0413 | 0.2947 | 0.2396 | 2.2660 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0060 | 0.9940 | 0.0031 | 0.9969 | 0.0027 | 0.9973 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0117 | 0.9884 | 0.0187 | 0.9815 | 0.0211 | 0.9792 | 0.0121 | 0.9880 | 0.1192 |
