# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 10.9418 | 0.3110 | 0.4745 | 2.2727 | 0.5900 |
| micro@1 | 9.0160 | 0.5753 | 0.3179 | 1.9665 | 0.7700 |
| mid@1 | 9.5705 | 0.5038 | 0.4784 | 2.1269 | 0.5730 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0134 | 0.9867 | 0.0117 | 0.9884 | 0.0017 | 0.9983 |
| micro@1 | 0.0224 | 0.9779 | 0.0201 | 0.9801 | 0.0036 | 0.9964 |
| mid@1 | 0.0159 | 0.9842 | 0.0102 | 0.9898 | 0.0059 | 0.9941 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0225 | 0.9778 | 0.0269 | 0.9735 | 0.0273 | 0.9731 | 0.0236 | 0.9767 | 0.1522 |
| micro@1 | 0.0291 | 0.9713 | 0.0294 | 0.9710 | 0.0289 | 0.9715 | 0.0232 | 0.9771 | 0.1661 |
| mid@1 | 0.0412 | 0.9596 | 0.0359 | 0.9647 | 0.0449 | 0.9561 | 0.0296 | 0.9708 | 0.1923 |
