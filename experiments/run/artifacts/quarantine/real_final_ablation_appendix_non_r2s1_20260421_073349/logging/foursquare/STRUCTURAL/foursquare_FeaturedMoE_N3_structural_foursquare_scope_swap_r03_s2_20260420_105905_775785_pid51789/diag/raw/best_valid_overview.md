# Diagnostic Overview

- split: best_valid
- feature_mode: runtime:unknown|ablation:none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 15.7769 | 0.1189 | 0.1268 | 2.5369 | 0.6997 |
| mid@1 | 15.6684 | 0.1455 | 0.1351 | 2.5196 | 0.6915 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0502 | 0.9510 | 0.0140 | 0.9861 | 0.0369 | 0.9638 |
| mid@1 | 0.0529 | 0.9485 | 0.0129 | 0.9872 | 0.0400 | 0.9608 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0621 | 0.9398 | 0.0595 | 0.9422 | 0.0605 | 0.9413 | 0.0571 | 0.9445 | 0.0788 |
| mid@1 | 0.0662 | 0.9360 | 0.0657 | 0.9364 | 0.0606 | 0.9412 | 0.0604 | 0.9414 | 0.0889 |
