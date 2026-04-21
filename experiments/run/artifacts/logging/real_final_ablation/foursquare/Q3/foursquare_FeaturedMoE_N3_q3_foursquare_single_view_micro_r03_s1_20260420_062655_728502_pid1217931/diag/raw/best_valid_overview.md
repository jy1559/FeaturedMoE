# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| micro@1 | 15.8245 | 0.1053 | 0.1107 | 2.5260 | 0.6325 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| micro@1 | 0.0586 | 0.9431 | 0.0137 | 0.9863 | 0.0450 | 0.9560 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| micro@1 | 0.0751 | 0.9277 | 0.0760 | 0.9268 | 0.0749 | 0.9278 | 0.0661 | 0.9360 | 0.0811 |
