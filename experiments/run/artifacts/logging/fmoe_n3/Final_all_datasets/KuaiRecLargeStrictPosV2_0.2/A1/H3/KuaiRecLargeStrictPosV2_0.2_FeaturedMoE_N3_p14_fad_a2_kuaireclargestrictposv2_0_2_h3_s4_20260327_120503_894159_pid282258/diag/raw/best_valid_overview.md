# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 10.8512 | 0.3254 | 0.6244 | 2.3225 | 0.5190 |
| micro@1 | 9.8566 | 0.4663 | 0.4099 | 2.1362 | 0.6674 |
| mid@1 | 9.1406 | 0.5593 | 0.5711 | 2.1997 | 0.6093 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0108 | 0.9892 | 0.0098 | 0.9903 | 0.0010 | 0.9990 |
| micro@1 | 0.0183 | 0.9818 | 0.0174 | 0.9828 | 0.0013 | 0.9987 |
| mid@1 | 0.0093 | 0.9908 | 0.0062 | 0.9938 | 0.0030 | 0.9970 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0172 | 0.9829 | 0.0203 | 0.9799 | 0.0207 | 0.9796 | 0.0176 | 0.9826 | 0.1619 |
| micro@1 | 0.0251 | 0.9752 | 0.0250 | 0.9753 | 0.0264 | 0.9739 | 0.0193 | 0.9809 | 0.1658 |
| mid@1 | 0.0217 | 0.9785 | 0.0211 | 0.9791 | 0.0214 | 0.9788 | 0.0138 | 0.9863 | 0.2102 |
