# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 11.4599 | 0.2171 | 0.5047 | 2.3665 | 0.0000 |
| micro@1 | 5.5867 | 1.0714 | 0.8179 | 1.7567 | 0.2785 |
| mid@1 | 11.3927 | 0.2309 | 0.4515 | 2.3577 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0024 | 0.9976 | 0.0024 | 0.9976 | 0.0000 | 1.0000 |
| micro@1 | 0.0011 | 0.9989 | 0.0011 | 0.9989 | 0.0000 | 1.0000 |
| mid@1 | 0.0017 | 0.9983 | 0.0017 | 0.9983 | 0.0000 | 1.0000 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0053 | 0.9947 | 0.0092 | 0.9909 | 0.0099 | 0.9901 | 0.0064 | 0.9936 | 0.1068 |
| micro@1 | 0.0019 | 0.9981 | 0.0032 | 0.9968 | 0.0014 | 0.9986 | 0.0084 | 0.9916 | 0.2483 |
| mid@1 | 0.0068 | 0.9932 | 0.0045 | 0.9955 | 0.0059 | 0.9941 | 0.0103 | 0.9898 | 0.1059 |
