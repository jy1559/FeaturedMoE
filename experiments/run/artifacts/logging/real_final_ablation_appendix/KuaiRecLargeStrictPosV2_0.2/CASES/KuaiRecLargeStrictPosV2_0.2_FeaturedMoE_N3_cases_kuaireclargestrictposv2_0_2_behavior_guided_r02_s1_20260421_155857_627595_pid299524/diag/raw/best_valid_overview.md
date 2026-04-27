# Diagnostic Overview

- split: best_valid
- feature_mode: runtime:unknown|ablation:none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 14.6690 | 0.3012 | 0.2435 | 2.5962 | 0.8652 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0180 | 0.9822 | 0.0056 | 0.9944 | 0.0128 | 0.9873 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0208 | 0.9794 | 0.0207 | 0.9795 | 0.0217 | 0.9785 | 0.0204 | 0.9798 | 0.1053 |
