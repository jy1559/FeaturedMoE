# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 6.5139 | 0.9177 | 0.9100 | 2.0129 | 0.0000 |
| micro@1 | 5.8752 | 1.0210 | 0.6873 | 1.6013 | 0.3909 |
| mid@1 | 4.5854 | 1.2716 | 0.9799 | 1.8134 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0149 | 0.9852 | 0.0014 | 0.9986 | 0.0124 | 0.9877 |
| micro@1 | 0.0115 | 0.9886 | 0.0016 | 0.9984 | 0.0117 | 0.9884 |
| mid@1 | 0.0107 | 0.9894 | 0.0007 | 0.9993 | 0.0129 | 0.9872 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0263 | 0.9741 | 0.0541 | 0.9473 | 0.0628 | 0.9391 | 0.0518 | 0.9495 | 0.3250 |
| micro@1 | 0.0359 | 0.9647 | 0.0469 | 0.9542 | 0.0446 | 0.9564 | 0.0252 | 0.9751 | 0.3180 |
| mid@1 | 0.0286 | 0.9718 | 0.0287 | 0.9717 | 0.0418 | 0.9591 | 0.0352 | 0.9654 | 0.4222 |
