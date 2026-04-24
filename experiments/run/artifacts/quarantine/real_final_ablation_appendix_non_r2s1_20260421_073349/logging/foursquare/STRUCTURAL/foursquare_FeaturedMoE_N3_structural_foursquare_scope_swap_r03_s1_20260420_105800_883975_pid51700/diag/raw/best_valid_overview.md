# Diagnostic Overview

- split: best_valid
- feature_mode: runtime:unknown|ablation:none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 15.8538 | 0.0960 | 0.1284 | 2.6021 | 0.6866 |
| mid@1 | 15.7956 | 0.1138 | 0.1294 | 2.5945 | 0.6742 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0379 | 0.9628 | 0.0096 | 0.9904 | 0.0284 | 0.9720 |
| mid@1 | 0.0402 | 0.9606 | 0.0091 | 0.9909 | 0.0311 | 0.9694 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0458 | 0.9552 | 0.0436 | 0.9573 | 0.0445 | 0.9564 | 0.0418 | 0.9590 | 0.0751 |
| mid@1 | 0.0487 | 0.9524 | 0.0494 | 0.9518 | 0.0460 | 0.9551 | 0.0454 | 0.9557 | 0.0824 |
