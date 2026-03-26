# Diagnostic Overview

- split: best_valid
- feature_mode: perturb:stage_mismatch@both#shift1

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 9.6134 | 0.4983 | 0.4354 | 1.9895 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0102 | 0.9899 | 0.0036 | 0.9964 | 0.0052 | 0.9948 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0356 | 0.9650 | 0.0321 | 0.9684 | 0.0387 | 0.9621 | 0.0458 | 0.9552 | 0.1883 |
