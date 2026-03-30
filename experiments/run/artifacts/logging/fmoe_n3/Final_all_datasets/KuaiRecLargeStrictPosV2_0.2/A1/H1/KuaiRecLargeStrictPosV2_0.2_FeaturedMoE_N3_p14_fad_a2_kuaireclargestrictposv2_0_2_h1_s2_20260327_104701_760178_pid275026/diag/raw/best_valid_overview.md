# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 11.4406 | 0.2211 | 0.3214 | 2.3224 | 0.7323 |
| micro@1 | 8.5069 | 0.6408 | 0.3763 | 1.9851 | 0.6877 |
| mid@1 | 9.2027 | 0.5513 | 0.4296 | 2.1447 | 0.6180 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0122 | 0.9879 | 0.0094 | 0.9906 | 0.0028 | 0.9972 |
| micro@1 | 0.0189 | 0.9812 | 0.0163 | 0.9839 | 0.0032 | 0.9969 |
| mid@1 | 0.0134 | 0.9867 | 0.0085 | 0.9915 | 0.0045 | 0.9955 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0196 | 0.9806 | 0.0236 | 0.9767 | 0.0216 | 0.9787 | 0.0225 | 0.9777 | 0.1189 |
| micro@1 | 0.0287 | 0.9717 | 0.0274 | 0.9730 | 0.0287 | 0.9717 | 0.0214 | 0.9788 | 0.1996 |
| mid@1 | 0.0410 | 0.9598 | 0.0357 | 0.9650 | 0.0434 | 0.9575 | 0.0205 | 0.9797 | 0.1849 |
