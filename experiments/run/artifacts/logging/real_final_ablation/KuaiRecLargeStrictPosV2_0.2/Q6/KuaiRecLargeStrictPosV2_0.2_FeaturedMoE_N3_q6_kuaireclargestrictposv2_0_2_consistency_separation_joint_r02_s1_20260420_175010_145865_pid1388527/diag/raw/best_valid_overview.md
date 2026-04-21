# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 14.6349 | 0.3054 | 0.2914 | 2.5554 | 0.8839 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0227 | 0.9775 | 0.0067 | 0.9934 | 0.0164 | 0.9838 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0266 | 0.9737 | 0.0271 | 0.9733 | 0.0273 | 0.9730 | 0.0250 | 0.9753 | 0.1033 |
