# Diagnostic Overview

- split: best_valid
- feature_mode: runtime:unknown|ablation:none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| micro@1 | 13.3134 | 0.7087 | 0.2021 | 2.2112 | 0.7688 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| micro@1 | 0.0628 | 0.9391 | 0.0072 | 0.9928 | 0.0574 | 0.9442 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| micro@1 | 0.0762 | 0.9267 | 0.0839 | 0.9195 | 0.0806 | 0.9226 | 0.0678 | 0.9344 | 0.1517 |
