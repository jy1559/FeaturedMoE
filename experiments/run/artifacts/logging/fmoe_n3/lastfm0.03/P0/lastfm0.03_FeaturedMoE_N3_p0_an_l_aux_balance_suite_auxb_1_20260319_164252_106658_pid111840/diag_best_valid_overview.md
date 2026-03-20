# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 17.6064 | 0.3687 | 0.1328 | 1.4653 | 0.0000 |
| micro@1 | 19.5844 | 0.1457 | 0.0840 | 1.8591 | 0.5378 |
| mid@1 | 18.5065 | 0.2841 | 0.0903 | 1.3651 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.1198 | 0.8871 | 0.0574 | 0.9442 | 0.0818 | 0.9215 |
| micro@1 | 0.0787 | 0.9243 | 0.0251 | 0.9752 | 0.0579 | 0.9438 |
| mid@1 | 0.1132 | 0.8930 | 0.0564 | 0.9452 | 0.0733 | 0.9293 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.1784 | 0.8366 | 0.1757 | 0.8389 | 0.1787 | 0.8364 | 0.1642 | 0.8485 | 0.1058 |
| micro@1 | 0.2028 | 0.8165 | 0.2216 | 0.8012 | 0.2492 | 0.7795 | 0.1695 | 0.8441 | 0.0720 |
| mid@1 | 0.1794 | 0.8358 | 0.1882 | 0.8285 | 0.2149 | 0.8066 | 0.1662 | 0.8469 | 0.0803 |
