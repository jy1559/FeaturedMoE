# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 11.4187 | 0.2256 | 0.2623 | 2.0739 | 0.0000 |
| micro@1 | 9.8792 | 0.4633 | 0.2668 | 1.5617 | 0.5176 |
| mid@1 | 9.2714 | 0.5425 | 0.4129 | 1.9368 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0252 | 0.9751 | 0.0129 | 0.9872 | 0.0130 | 0.9870 |
| micro@1 | 0.0330 | 0.9675 | 0.0144 | 0.9857 | 0.0207 | 0.9795 |
| mid@1 | 0.0201 | 0.9801 | 0.0070 | 0.9930 | 0.0112 | 0.9889 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0345 | 0.9661 | 0.0426 | 0.9583 | 0.0442 | 0.9568 | 0.0365 | 0.9642 | 0.1190 |
| micro@1 | 0.0559 | 0.9456 | 0.0652 | 0.9369 | 0.0656 | 0.9366 | 0.0609 | 0.9409 | 0.1798 |
| mid@1 | 0.0337 | 0.9669 | 0.0325 | 0.9680 | 0.0387 | 0.9620 | 0.0421 | 0.9587 | 0.2118 |
