# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 11.4436 | 0.2205 | 0.3186 | 2.3175 | 0.7313 |
| micro@1 | 8.3312 | 0.6636 | 0.3632 | 1.9463 | 0.6750 |
| mid@1 | 9.2852 | 0.5407 | 0.4340 | 2.1487 | 0.6060 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0127 | 0.9874 | 0.0096 | 0.9904 | 0.0031 | 0.9969 |
| micro@1 | 0.0208 | 0.9795 | 0.0178 | 0.9824 | 0.0037 | 0.9964 |
| mid@1 | 0.0135 | 0.9866 | 0.0084 | 0.9917 | 0.0048 | 0.9952 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0201 | 0.9801 | 0.0242 | 0.9761 | 0.0220 | 0.9783 | 0.0233 | 0.9770 | 0.1192 |
| micro@1 | 0.0338 | 0.9668 | 0.0323 | 0.9682 | 0.0340 | 0.9666 | 0.0233 | 0.9770 | 0.1967 |
| mid@1 | 0.0421 | 0.9588 | 0.0360 | 0.9646 | 0.0445 | 0.9564 | 0.0216 | 0.9786 | 0.1845 |
