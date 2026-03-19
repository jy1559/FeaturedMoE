# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 5.9198 | 0.1164 | 0.3095 | 1.3100 | 0.0000 |
| micro@1 | 4.6417 | 0.5410 | 0.3359 | 1.1034 | 0.5188 |
| mid@1 | 5.1143 | 0.4162 | 0.3349 | 1.1994 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0316 | 0.9689 | 0.0078 | 0.9922 | 0.0255 | 0.9748 |
| micro@1 | 0.0226 | 0.9776 | 0.0036 | 0.9964 | 0.0168 | 0.9834 |
| mid@1 | 0.0283 | 0.9721 | 0.0108 | 0.9892 | 0.0143 | 0.9858 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0384 | 0.9623 | 0.0516 | 0.9497 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1031 |
| micro@1 | 0.0354 | 0.9652 | 0.0421 | 0.9587 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1603 |
| mid@1 | 0.0581 | 0.9435 | 0.0284 | 0.9720 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1438 |
