# Diagnostic Overview

- split: best_valid
- feature_mode: perturb:family_permute@train:focus#shift1

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 11.7271 | 0.1526 | 0.2873 | 2.2229 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0093 | 0.9907 | 0.0035 | 0.9966 | 0.0053 | 0.9948 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0167 | 0.9834 | 0.0291 | 0.9713 | 0.0272 | 0.9731 | 0.0225 | 0.9778 | 0.1192 |
