# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 5.8611 | 0.1539 | 0.3946 | 1.4541 | 0.0000 |
| micro@1 | 5.8683 | 0.1498 | 0.2658 | 1.3737 | 0.5538 |
| mid@1 | 5.8519 | 0.1591 | 0.3807 | 1.3910 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0220 | 0.9783 | 0.0052 | 0.9948 | 0.0162 | 0.9839 |
| micro@1 | 0.0220 | 0.9782 | 0.0069 | 0.9931 | 0.0158 | 0.9844 |
| mid@1 | 0.0170 | 0.9832 | 0.0049 | 0.9952 | 0.0116 | 0.9885 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1037 |
| micro@1 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1103 |
| mid@1 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1124 |
