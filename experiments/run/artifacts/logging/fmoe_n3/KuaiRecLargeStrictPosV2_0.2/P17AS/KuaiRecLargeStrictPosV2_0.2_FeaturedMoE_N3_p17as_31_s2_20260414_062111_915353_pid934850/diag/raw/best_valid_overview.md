# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 15.5382 | 0.1724 | 0.7420 | 2.7544 | 0.0000 |
| mid@1 | 8.8361 | 0.9004 | 0.8283 | 2.4685 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0003 | 0.9997 | 0.0003 | 0.9997 | 0.0000 | 1.0000 |
| mid@1 | 0.0015 | 0.9985 | 0.0005 | 0.9995 | 0.0015 | 0.9985 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0005 | 0.9995 | 0.0009 | 0.9991 | 0.0009 | 0.9991 | 0.0009 | 0.9991 | 0.0749 |
| mid@1 | 0.0101 | 0.9899 | 0.0071 | 0.9929 | 0.0056 | 0.9944 | 0.0179 | 0.9822 | 0.2771 |
