# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 11.4987 | 0.2088 | 0.2529 | 2.3306 | 0.6992 |
| micro@1 | 8.5736 | 0.6322 | 0.3157 | 1.9343 | 0.6668 |
| mid@1 | 9.3498 | 0.5324 | 0.4472 | 2.1321 | 0.5565 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0131 | 0.9870 | 0.0092 | 0.9908 | 0.0039 | 0.9961 |
| micro@1 | 0.0201 | 0.9801 | 0.0164 | 0.9837 | 0.0039 | 0.9962 |
| mid@1 | 0.0156 | 0.9845 | 0.0097 | 0.9904 | 0.0057 | 0.9943 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0209 | 0.9793 | 0.0249 | 0.9754 | 0.0214 | 0.9788 | 0.0243 | 0.9760 | 0.1074 |
| micro@1 | 0.0327 | 0.9678 | 0.0315 | 0.9689 | 0.0323 | 0.9682 | 0.0230 | 0.9772 | 0.2010 |
| mid@1 | 0.0482 | 0.9529 | 0.0383 | 0.9624 | 0.0528 | 0.9486 | 0.0293 | 0.9711 | 0.1837 |
