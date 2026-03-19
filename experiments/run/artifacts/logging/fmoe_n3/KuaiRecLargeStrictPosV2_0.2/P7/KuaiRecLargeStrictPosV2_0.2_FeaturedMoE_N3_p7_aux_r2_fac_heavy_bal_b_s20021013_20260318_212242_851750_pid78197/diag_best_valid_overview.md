# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 11.4284 | 0.2236 | 0.1478 | 1.6161 | 0.0000 |
| micro@1 | 10.2974 | 0.4066 | 0.1592 | 1.4162 | 0.6025 |
| mid@1 | 9.6826 | 0.4892 | 0.2015 | 1.6358 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0506 | 0.9507 | 0.0454 | 0.9557 | 0.0064 | 0.9936 |
| micro@1 | 0.0496 | 0.9516 | 0.0434 | 0.9576 | 0.0103 | 0.9897 |
| mid@1 | 0.0378 | 0.9630 | 0.0292 | 0.9712 | 0.0101 | 0.9899 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0648 | 0.9372 | 0.0787 | 0.9244 | 0.0844 | 0.9191 | 0.0706 | 0.9318 | 0.1135 |
| micro@1 | 0.0795 | 0.9236 | 0.0966 | 0.9079 | 0.1009 | 0.9041 | 0.0847 | 0.9188 | 0.1415 |
| mid@1 | 0.0815 | 0.9217 | 0.0763 | 0.9265 | 0.0861 | 0.9175 | 0.0697 | 0.9327 | 0.1534 |
