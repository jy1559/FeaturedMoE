# Diagnostic Overview

- split: best_valid
- feature_mode: runtime:unknown|ablation:perturb:role_swap@both#shift1

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 11.8993 | 0.0920 | 0.7646 | 2.4540 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0008 | 0.9992 | 0.0005 | 0.9995 | 0.0003 | 0.9997 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0020 | 0.9980 | 0.0022 | 0.9978 | 0.0024 | 0.9976 | 0.0009 | 0.9991 | 0.0996 |
