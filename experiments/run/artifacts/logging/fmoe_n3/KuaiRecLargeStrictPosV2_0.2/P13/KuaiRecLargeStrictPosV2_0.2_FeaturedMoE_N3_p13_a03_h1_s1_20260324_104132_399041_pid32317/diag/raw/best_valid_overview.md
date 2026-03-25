# Diagnostic Overview

- split: best_valid
- feature_mode: perturb:shuffle@eval#shift1

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 11.9705 | 0.0496 | 0.1654 | 2.3951 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0067 | 0.9934 | 0.0049 | 0.9951 | 0.0018 | 0.9982 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0107 | 0.9893 | 0.0141 | 0.9860 | 0.0154 | 0.9848 | 0.0119 | 0.9882 | 0.0968 |
