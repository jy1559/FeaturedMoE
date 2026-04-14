# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 15.5382 | 0.1724 | 0.7420 | 2.7544 | 0.0000 |
| mid@1 | 7.6515 | 1.0445 | 0.8271 | 2.3772 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0003 | 0.9997 | 0.0003 | 0.9997 | 0.0000 | 1.0000 |
| mid@1 | 0.0014 | 0.9986 | 0.0005 | 0.9995 | 0.0013 | 0.9987 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0005 | 0.9995 | 0.0009 | 0.9991 | 0.0009 | 0.9991 | 0.0009 | 0.9991 | 0.0749 |
| mid@1 | 0.0114 | 0.9887 | 0.0072 | 0.9928 | 0.0047 | 0.9953 | 0.0243 | 0.9760 | 0.3120 |
