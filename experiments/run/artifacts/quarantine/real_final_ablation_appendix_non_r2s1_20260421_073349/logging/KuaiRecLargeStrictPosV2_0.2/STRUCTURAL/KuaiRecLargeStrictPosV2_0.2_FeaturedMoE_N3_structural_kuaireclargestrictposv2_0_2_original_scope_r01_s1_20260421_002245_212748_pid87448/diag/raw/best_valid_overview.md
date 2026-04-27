# Diagnostic Overview

- split: best_valid
- feature_mode: runtime:unknown|ablation:none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 13.9329 | 0.6599 | 0.1688 | 2.1938 | 0.8938 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0606 | 0.9412 | 0.0148 | 0.9854 | 0.0469 | 0.9542 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0785 | 0.9245 | 0.0814 | 0.9218 | 0.0830 | 0.9203 | 0.0737 | 0.9289 | 0.1208 |
