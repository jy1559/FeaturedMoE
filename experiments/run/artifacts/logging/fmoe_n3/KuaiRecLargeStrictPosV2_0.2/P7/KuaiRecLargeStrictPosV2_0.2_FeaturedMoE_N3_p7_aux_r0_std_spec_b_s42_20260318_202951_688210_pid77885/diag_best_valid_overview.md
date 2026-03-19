# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 3.8281 | 1.4611 | 0.3901 | 0.7437 | 0.0000 |
| micro@1 | 4.0813 | 1.3929 | 0.4381 | 0.6491 | 0.3852 |
| mid@1 | 1.7430 | 2.4258 | 0.7594 | 0.5070 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0558 | 0.9458 | 0.0317 | 0.9688 | 0.0303 | 0.9701 |
| micro@1 | 0.0458 | 0.9552 | 0.0235 | 0.9768 | 0.0347 | 0.9659 |
| mid@1 | 0.0098 | 0.9902 | 0.0055 | 0.9945 | 0.0145 | 0.9857 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0718 | 0.9307 | 0.0786 | 0.9244 | 0.0851 | 0.9184 | 0.0699 | 0.9325 | 0.3615 |
| micro@1 | 0.0786 | 0.9244 | 0.0922 | 0.9119 | 0.0958 | 0.9086 | 0.0706 | 0.9319 | 0.4335 |
| mid@1 | 0.0242 | 0.9761 | 0.0194 | 0.9808 | 0.0144 | 0.9857 | 0.0604 | 0.9414 | 0.7761 |
