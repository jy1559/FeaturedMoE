# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 11.4873 | 0.2113 | 0.2364 | 2.3265 | 0.5175 |
| micro@1 | 11.4006 | 0.2293 | 0.2429 | 2.2585 | 0.5995 |
| mid@1 | 11.5820 | 0.1900 | 0.2878 | 2.0778 | 0.5049 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0179 | 0.9823 | 0.0150 | 0.9851 | 0.0025 | 0.9975 |
| micro@1 | 0.0258 | 0.9745 | 0.0247 | 0.9756 | 0.0013 | 0.9987 |
| mid@1 | 0.0313 | 0.9692 | 0.0245 | 0.9758 | 0.0068 | 0.9932 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0285 | 0.9719 | 0.0286 | 0.9718 | 0.0277 | 0.9727 | 0.0277 | 0.9727 | 0.1060 |
| micro@1 | 0.0460 | 0.9550 | 0.0343 | 0.9663 | 0.0489 | 0.9523 | 0.0367 | 0.9639 | 0.1158 |
| mid@1 | 0.0944 | 0.9100 | 0.0636 | 0.9384 | 0.0909 | 0.9132 | 0.0636 | 0.9384 | 0.1245 |
