# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 11.6641 | 0.1697 | 0.2591 | 2.1052 | 0.0000 |
| micro@1 | 9.8369 | 0.4689 | 0.3096 | 1.7454 | 0.4852 |
| mid@1 | 10.1558 | 0.4261 | 0.3180 | 1.9894 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0219 | 0.9784 | 0.0107 | 0.9894 | 0.0107 | 0.9894 |
| micro@1 | 0.0248 | 0.9755 | 0.0103 | 0.9898 | 0.0143 | 0.9858 |
| mid@1 | 0.0189 | 0.9813 | 0.0064 | 0.9936 | 0.0105 | 0.9896 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0301 | 0.9703 | 0.0368 | 0.9639 | 0.0384 | 0.9624 | 0.0311 | 0.9694 | 0.1116 |
| micro@1 | 0.0417 | 0.9591 | 0.0498 | 0.9515 | 0.0499 | 0.9513 | 0.0495 | 0.9517 | 0.1737 |
| mid@1 | 0.0317 | 0.9688 | 0.0309 | 0.9696 | 0.0388 | 0.9619 | 0.0390 | 0.9618 | 0.1732 |
