# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 5.8663 | 0.1509 | 0.3455 | 1.3838 | 0.0000 |
| micro@1 | 5.5768 | 0.2755 | 0.3168 | 1.2150 | 0.4977 |
| mid@1 | 5.6790 | 0.2378 | 0.3262 | 1.3844 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0254 | 0.9749 | 0.0067 | 0.9933 | 0.0187 | 0.9815 |
| micro@1 | 0.0264 | 0.9739 | 0.0070 | 0.9930 | 0.0192 | 0.9810 |
| mid@1 | 0.0174 | 0.9828 | 0.0079 | 0.9922 | 0.0099 | 0.9902 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0000 | 1.0000 | 0.0382 | 0.9625 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1000 |
| micro@1 | 0.0000 | 1.0000 | 0.0503 | 0.9510 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1471 |
| mid@1 | 0.0000 | 1.0000 | 0.0216 | 0.9786 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1240 |
