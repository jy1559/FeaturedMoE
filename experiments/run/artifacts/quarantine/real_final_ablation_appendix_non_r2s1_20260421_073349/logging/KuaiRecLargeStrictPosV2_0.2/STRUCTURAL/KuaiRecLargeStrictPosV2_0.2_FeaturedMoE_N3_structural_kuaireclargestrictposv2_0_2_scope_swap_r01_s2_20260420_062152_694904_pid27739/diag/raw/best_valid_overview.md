# Diagnostic Overview

- split: best_valid
- feature_mode: runtime:unknown|ablation:none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 15.2860 | 0.5553 | 0.1818 | 2.2628 | 0.8954 |
| mid@1 | 8.8281 | 1.1249 | 0.3821 | 1.8740 | 0.7958 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0610 | 0.9408 | 0.0145 | 0.9856 | 0.0473 | 0.9538 |
| mid@1 | 0.0670 | 0.9352 | 0.0179 | 0.9823 | 0.0524 | 0.9489 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0775 | 0.9254 | 0.0796 | 0.9235 | 0.0805 | 0.9226 | 0.0739 | 0.9287 | 0.1140 |
| mid@1 | 0.0920 | 0.9121 | 0.0906 | 0.9134 | 0.0860 | 0.9176 | 0.0771 | 0.9258 | 0.2620 |
