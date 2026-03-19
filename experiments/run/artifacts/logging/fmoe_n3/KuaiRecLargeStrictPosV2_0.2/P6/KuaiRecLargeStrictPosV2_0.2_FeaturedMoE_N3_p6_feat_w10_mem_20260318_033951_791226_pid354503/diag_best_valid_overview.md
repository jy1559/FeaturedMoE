# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 2.9479 | 0.1329 | 0.4816 | 0.7034 | 0.0000 |
| micro@1 | 2.9994 | 0.0140 | 0.3510 | 0.8787 | 0.5222 |
| mid@1 | 2.7177 | 0.3223 | 0.5639 | 0.4555 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0272 | 0.9732 | 0.0000 | 1.0000 | 0.0272 | 0.9732 |
| micro@1 | 0.0131 | 0.9870 | -0.0000 | 1.0000 | 0.0131 | 0.9870 |
| mid@1 | 0.0169 | 0.9833 | 0.0000 | 1.0000 | 0.0169 | 0.9833 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0944 |
| micro@1 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0924 |
| mid@1 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1265 |
