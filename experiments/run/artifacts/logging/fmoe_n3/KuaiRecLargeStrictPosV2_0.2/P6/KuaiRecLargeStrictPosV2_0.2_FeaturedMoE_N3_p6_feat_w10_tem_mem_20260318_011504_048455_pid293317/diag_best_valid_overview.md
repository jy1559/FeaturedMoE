# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 5.9369 | 0.1031 | 0.3096 | 1.4243 | 0.0000 |
| micro@1 | 5.4529 | 0.3168 | 0.2950 | 1.2558 | 0.4333 |
| mid@1 | 4.6212 | 0.5462 | 0.5298 | 1.1060 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0217 | 0.9785 | 0.0065 | 0.9935 | 0.0157 | 0.9844 |
| micro@1 | 0.0205 | 0.9797 | 0.0058 | 0.9942 | 0.0154 | 0.9847 |
| mid@1 | 0.0221 | 0.9782 | 0.0048 | 0.9952 | 0.0137 | 0.9864 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0245 | 0.9758 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0938 |
| micro@1 | 0.0323 | 0.9682 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1163 |
| mid@1 | 0.0386 | 0.9621 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1847 |
