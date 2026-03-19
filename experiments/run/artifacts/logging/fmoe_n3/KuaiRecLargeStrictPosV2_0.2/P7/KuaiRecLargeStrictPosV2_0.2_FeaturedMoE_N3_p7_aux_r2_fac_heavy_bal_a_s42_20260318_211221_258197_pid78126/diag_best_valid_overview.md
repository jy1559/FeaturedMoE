# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 11.4739 | 0.2141 | 0.1546 | 1.6325 | 0.0000 |
| micro@1 | 9.9254 | 0.4572 | 0.2187 | 1.4031 | 0.5467 |
| mid@1 | 8.9340 | 0.5858 | 0.1919 | 1.5834 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0505 | 0.9508 | 0.0459 | 0.9551 | 0.0059 | 0.9941 |
| micro@1 | 0.0477 | 0.9534 | 0.0420 | 0.9589 | 0.0096 | 0.9905 |
| mid@1 | 0.0351 | 0.9656 | 0.0276 | 0.9727 | 0.0086 | 0.9914 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0644 | 0.9376 | 0.0767 | 0.9262 | 0.0829 | 0.9204 | 0.0694 | 0.9330 | 0.1139 |
| micro@1 | 0.0767 | 0.9262 | 0.0920 | 0.9121 | 0.0971 | 0.9075 | 0.0815 | 0.9218 | 0.1518 |
| mid@1 | 0.0766 | 0.9263 | 0.0709 | 0.9315 | 0.0781 | 0.9249 | 0.0673 | 0.9349 | 0.1742 |
