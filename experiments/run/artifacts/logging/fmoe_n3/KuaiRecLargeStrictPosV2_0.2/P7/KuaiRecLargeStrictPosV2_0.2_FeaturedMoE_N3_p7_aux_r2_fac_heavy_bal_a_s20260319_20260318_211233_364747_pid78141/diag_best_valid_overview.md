# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 11.3144 | 0.2462 | 0.1698 | 1.7085 | 0.0000 |
| micro@1 | 9.6786 | 0.4897 | 0.2685 | 1.5066 | 0.5728 |
| mid@1 | 8.6100 | 0.6275 | 0.1922 | 1.5463 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0479 | 0.9532 | 0.0438 | 0.9571 | 0.0052 | 0.9948 |
| micro@1 | 0.0470 | 0.9541 | 0.0421 | 0.9588 | 0.0071 | 0.9929 |
| mid@1 | 0.0328 | 0.9678 | 0.0263 | 0.9740 | 0.0075 | 0.9926 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0608 | 0.9410 | 0.0714 | 0.9311 | 0.0774 | 0.9255 | 0.0646 | 0.9374 | 0.1195 |
| micro@1 | 0.0767 | 0.9262 | 0.0919 | 0.9122 | 0.0939 | 0.9104 | 0.0804 | 0.9227 | 0.1551 |
| mid@1 | 0.0730 | 0.9296 | 0.0669 | 0.9353 | 0.0750 | 0.9277 | 0.0651 | 0.9370 | 0.1733 |
