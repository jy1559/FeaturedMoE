# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 11.4906 | 0.2106 | 0.2145 | 1.9854 | 0.0000 |
| micro@1 | 11.4951 | 0.2096 | 0.1411 | 1.9179 | 0.5280 |
| mid@1 | 9.6186 | 0.4976 | 0.3294 | 1.7187 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0382 | 0.9626 | 0.0159 | 0.9842 | 0.0255 | 0.9749 |
| micro@1 | 0.0337 | 0.9669 | 0.0132 | 0.9869 | 0.0215 | 0.9787 |
| mid@1 | 0.0337 | 0.9668 | 0.0181 | 0.9821 | 0.0175 | 0.9827 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0463 | 0.9548 | 0.0513 | 0.9500 | 0.0558 | 0.9458 | 0.0494 | 0.9518 | 0.1124 |
| micro@1 | 0.0609 | 0.9410 | 0.0710 | 0.9314 | 0.0789 | 0.9241 | 0.0503 | 0.9509 | 0.1097 |
| mid@1 | 0.0516 | 0.9497 | 0.0497 | 0.9515 | 0.0596 | 0.9421 | 0.0578 | 0.9438 | 0.1905 |
