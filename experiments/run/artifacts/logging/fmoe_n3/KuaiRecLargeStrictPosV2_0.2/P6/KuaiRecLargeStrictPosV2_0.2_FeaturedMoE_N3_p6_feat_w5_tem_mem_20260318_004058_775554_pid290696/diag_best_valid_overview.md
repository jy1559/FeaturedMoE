# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 5.9221 | 0.1147 | 0.2860 | 1.4363 | 0.0000 |
| micro@1 | 5.4088 | 0.3306 | 0.3090 | 1.2025 | 0.4407 |
| mid@1 | 4.5618 | 0.5615 | 0.5252 | 1.0424 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0234 | 0.9769 | 0.0069 | 0.9931 | 0.0176 | 0.9826 |
| micro@1 | 0.0221 | 0.9782 | 0.0063 | 0.9937 | 0.0168 | 0.9833 |
| mid@1 | 0.0252 | 0.9751 | 0.0047 | 0.9953 | 0.0168 | 0.9833 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0268 | 0.9736 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0958 |
| micro@1 | 0.0335 | 0.9670 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1200 |
| mid@1 | 0.0442 | 0.9568 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1845 |
