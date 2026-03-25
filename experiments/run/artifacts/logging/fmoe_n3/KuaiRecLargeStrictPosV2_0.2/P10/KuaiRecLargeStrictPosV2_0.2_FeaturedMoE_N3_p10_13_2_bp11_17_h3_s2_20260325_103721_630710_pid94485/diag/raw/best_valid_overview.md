# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 9.2781 | 0.5416 | 0.3671 | 1.9070 | 0.0000 |
| micro@1 | 8.5420 | 0.6363 | 0.6328 | 2.0469 | 0.4557 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0169 | 0.9832 | 0.0080 | 0.9920 | 0.0086 | 0.9915 |
| micro@1 | 0.0020 | 0.9980 | 0.0004 | 0.9996 | 0.0016 | 0.9984 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0287 | 0.9717 | 0.0496 | 0.9516 | 0.0485 | 0.9527 | 0.0407 | 0.9601 | 0.1920 |
| micro@1 | 0.0063 | 0.9937 | 0.0086 | 0.9914 | 0.0071 | 0.9929 | 0.0064 | 0.9936 | 0.2145 |
