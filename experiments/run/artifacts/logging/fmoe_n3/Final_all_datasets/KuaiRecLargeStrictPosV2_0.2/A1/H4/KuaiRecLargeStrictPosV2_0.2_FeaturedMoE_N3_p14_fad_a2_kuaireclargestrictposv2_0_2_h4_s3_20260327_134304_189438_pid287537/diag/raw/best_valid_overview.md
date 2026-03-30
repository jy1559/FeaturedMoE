# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 10.0341 | 0.4426 | 0.2986 | 2.2243 | 0.7046 |
| micro@1 | 10.5486 | 0.3709 | 0.2854 | 2.1419 | 0.7487 |
| mid@1 | 9.6253 | 0.4967 | 0.5011 | 2.0959 | 0.4610 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0138 | 0.9863 | 0.0032 | 0.9968 | 0.0117 | 0.9884 |
| micro@1 | 0.0114 | 0.9887 | 0.0087 | 0.9913 | 0.0032 | 0.9968 |
| mid@1 | 0.0222 | 0.9781 | 0.0082 | 0.9919 | 0.0134 | 0.9867 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0210 | 0.9793 | 0.0361 | 0.9645 | 0.0368 | 0.9639 | 0.0348 | 0.9658 | 0.1430 |
| micro@1 | 0.0253 | 0.9750 | 0.0243 | 0.9760 | 0.0253 | 0.9751 | 0.0148 | 0.9853 | 0.1342 |
| mid@1 | 0.0630 | 0.9389 | 0.0506 | 0.9507 | 0.0760 | 0.9268 | 0.0628 | 0.9391 | 0.1413 |
