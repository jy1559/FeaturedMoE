# Diagnostic Overview

- split: best_valid
- feature_mode: runtime:unknown|ablation:perturb:family_permute@eval:memory#shift1

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 7.3624 | 0.7937 | 0.7552 | 2.0967 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0222 | 0.9780 | 0.0038 | 0.9962 | 0.0212 | 0.9791 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0439 | 0.9571 | 0.0449 | 0.9560 | 0.0559 | 0.9457 | 0.0438 | 0.9571 | 0.2285 |
