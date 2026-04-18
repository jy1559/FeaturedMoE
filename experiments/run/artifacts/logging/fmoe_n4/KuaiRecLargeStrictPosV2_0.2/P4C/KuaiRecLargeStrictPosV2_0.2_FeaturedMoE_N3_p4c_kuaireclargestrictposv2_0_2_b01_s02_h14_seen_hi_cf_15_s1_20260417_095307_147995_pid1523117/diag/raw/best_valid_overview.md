# Diagnostic Overview

- split: best_valid
- feature_mode: runtime:unknown|ablation:perturb:batch_permute@train#shift1

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 9.1829 | 0.5539 | 0.9561 | 2.3201 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0030 | 0.9970 | 0.0007 | 0.9993 | 0.0022 | 0.9978 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0056 | 0.9944 | 0.0106 | 0.9895 | 0.0121 | 0.9880 | 0.0066 | 0.9934 | 0.2175 |
