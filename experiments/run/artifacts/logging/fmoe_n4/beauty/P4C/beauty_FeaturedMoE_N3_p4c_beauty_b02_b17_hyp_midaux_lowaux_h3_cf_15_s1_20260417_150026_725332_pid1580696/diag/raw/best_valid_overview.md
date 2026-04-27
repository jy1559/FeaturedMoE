# Diagnostic Overview

- split: best_valid
- feature_mode: runtime:unknown|ablation:perturb:batch_permute@train#shift1

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 6.6152 | 0.4575 | 0.9956 | 1.9367 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0008 | 0.9992 | 0.0008 | 0.9992 | 0.0001 | 0.9999 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0013 | 0.9987 | 0.0011 | 0.9989 | 0.0018 | 0.9982 | 0.0012 | 0.9988 | 0.2155 |
