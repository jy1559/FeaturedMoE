# Diagnostic Overview

- split: best_valid
- feature_mode: runtime:unknown|ablation:none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 11.9416 | 0.0699 | 0.1455 | 2.3457 | 0.6819 |
| mid@1 | 11.9602 | 0.0577 | 0.1389 | 2.3525 | 0.6861 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0332 | 0.9674 | 0.0109 | 0.9892 | 0.0225 | 0.9777 |
| mid@1 | 0.0322 | 0.9683 | 0.0131 | 0.9869 | 0.0192 | 0.9810 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0385 | 0.9623 | 0.0382 | 0.9625 | 0.0375 | 0.9632 | 0.0362 | 0.9645 | 0.0929 |
| mid@1 | 0.0364 | 0.9643 | 0.0354 | 0.9652 | 0.0363 | 0.9643 | 0.0360 | 0.9646 | 0.0905 |
