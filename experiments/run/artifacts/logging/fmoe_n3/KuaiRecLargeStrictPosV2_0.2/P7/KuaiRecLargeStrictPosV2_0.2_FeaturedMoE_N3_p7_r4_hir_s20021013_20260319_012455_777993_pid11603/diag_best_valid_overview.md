# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 11.5001 | 0.2085 | 0.1915 | 1.8937 | 0.0000 |
| micro@1 | 9.3783 | 0.5287 | 0.2156 | 1.1513 | 0.5118 |
| mid@1 | 11.2734 | 0.2539 | 0.2758 | 1.9914 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0545 | 0.9469 | 0.0314 | 0.9691 | 0.0240 | 0.9763 |
| micro@1 | 0.0592 | 0.9425 | 0.0365 | 0.9642 | 0.0272 | 0.9731 |
| mid@1 | 0.0332 | 0.9673 | 0.0164 | 0.9837 | 0.0184 | 0.9818 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0668 | 0.9354 | 0.0740 | 0.9287 | 0.0790 | 0.9240 | 0.0698 | 0.9326 | 0.1142 |
| micro@1 | 0.0981 | 0.9065 | 0.1152 | 0.8912 | 0.1200 | 0.8869 | 0.0951 | 0.9093 | 0.1897 |
| mid@1 | 0.0499 | 0.9514 | 0.0495 | 0.9517 | 0.0568 | 0.9448 | 0.0558 | 0.9458 | 0.1274 |
