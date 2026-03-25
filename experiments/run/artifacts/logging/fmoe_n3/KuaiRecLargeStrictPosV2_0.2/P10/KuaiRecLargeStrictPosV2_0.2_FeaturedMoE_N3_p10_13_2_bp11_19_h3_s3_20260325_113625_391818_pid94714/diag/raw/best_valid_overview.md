# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 11.8028 | 0.1293 | 0.2165 | 2.2315 | 0.0000 |
| micro@1 | 9.1427 | 0.5590 | 0.3267 | 1.9493 | 0.6752 |
| mid@1 | 9.5921 | 0.5010 | 0.2360 | 1.8301 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0094 | 0.9906 | 0.0043 | 0.9957 | 0.0047 | 0.9953 |
| micro@1 | 0.0032 | 0.9968 | 0.0013 | 0.9987 | 0.0021 | 0.9979 |
| mid@1 | 0.0099 | 0.9901 | 0.0034 | 0.9966 | 0.0046 | 0.9954 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0166 | 0.9835 | 0.0299 | 0.9706 | 0.0297 | 0.9707 | 0.0227 | 0.9775 | 0.1109 |
| micro@1 | 0.0089 | 0.9911 | 0.0123 | 0.9878 | 0.0122 | 0.9879 | 0.0105 | 0.9895 | 0.1801 |
| mid@1 | 0.0418 | 0.9591 | 0.0284 | 0.9720 | 0.0382 | 0.9625 | 0.0639 | 0.9381 | 0.1463 |
