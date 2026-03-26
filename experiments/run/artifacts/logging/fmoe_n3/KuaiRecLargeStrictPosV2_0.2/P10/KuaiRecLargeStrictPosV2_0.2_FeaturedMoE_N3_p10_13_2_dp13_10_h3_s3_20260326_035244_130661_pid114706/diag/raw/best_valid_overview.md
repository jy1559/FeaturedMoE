# Diagnostic Overview

- split: best_valid
- feature_mode: perturb:family_permute@train:tempo#shift1

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 11.8360 | 0.1177 | 0.2545 | 2.2372 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0083 | 0.9917 | 0.0054 | 0.9946 | 0.0027 | 0.9973 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0220 | 0.9782 | 0.0225 | 0.9777 | 0.0222 | 0.9781 | 0.0085 | 0.9916 | 0.1037 |
