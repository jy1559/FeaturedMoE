# Diagnostic Overview

- split: best_valid
- feature_mode: perturb:shuffle@eval#shift1

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 11.9722 | 0.0482 | 0.4067 | 2.4652 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0015 | 0.9985 | 0.0012 | 0.9988 | 0.0003 | 0.9997 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0016 | 0.9984 | 0.0023 | 0.9977 | 0.0025 | 0.9975 | 0.0018 | 0.9982 | 0.0920 |
