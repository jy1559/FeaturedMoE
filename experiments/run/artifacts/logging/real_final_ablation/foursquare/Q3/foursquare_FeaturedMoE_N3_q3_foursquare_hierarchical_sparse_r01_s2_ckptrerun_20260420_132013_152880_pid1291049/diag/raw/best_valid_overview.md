# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 15.8607 | 0.0937 | 0.0897 | 2.5084 | 0.7188 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0673 | 0.9349 | 0.0178 | 0.9824 | 0.0496 | 0.9516 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0786 | 0.9244 | 0.0761 | 0.9268 | 0.0763 | 0.9265 | 0.0713 | 0.9311 | 0.0746 |
