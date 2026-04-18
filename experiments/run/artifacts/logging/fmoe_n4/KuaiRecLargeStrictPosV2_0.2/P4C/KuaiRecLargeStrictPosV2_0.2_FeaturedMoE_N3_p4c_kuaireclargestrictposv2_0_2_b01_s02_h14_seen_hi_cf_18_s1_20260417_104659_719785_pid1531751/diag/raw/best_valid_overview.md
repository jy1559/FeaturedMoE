# Diagnostic Overview

- split: best_valid
- feature_mode: runtime:unknown|ablation:perturb:position_shift@both#shift1

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 9.0725 | 0.5680 | 0.9372 | 2.3172 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0034 | 0.9966 | 0.0005 | 0.9995 | 0.0026 | 0.9974 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0122 | 0.9879 | 0.0168 | 0.9833 | 0.0176 | 0.9826 | 0.0147 | 0.9854 | 0.2349 |
