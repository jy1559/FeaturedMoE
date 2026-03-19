# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 9.4845 | 0.5150 | 0.3063 | 1.6118 | 0.0000 |
| micro@1 | 9.5532 | 0.5061 | 0.2091 | 1.2256 | 0.4928 |
| mid@1 | 9.8441 | 0.4680 | 0.2394 | 1.4668 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.1017 | 0.9033 | 0.0347 | 0.9659 | 0.1087 | 0.8971 |
| micro@1 | 0.0533 | 0.9481 | 0.0216 | 0.9786 | 0.0544 | 0.9472 |
| mid@1 | 0.0646 | 0.9375 | 0.0267 | 0.9737 | 0.0936 | 0.9108 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.1198 | 0.8871 | 0.1378 | 0.8712 | 0.1526 | 0.8584 | 0.1382 | 0.8709 | 0.1681 |
| micro@1 | 0.0827 | 0.9206 | 0.0983 | 0.9064 | 0.1069 | 0.8987 | 0.0844 | 0.9191 | 0.1603 |
| mid@1 | 0.0944 | 0.9099 | 0.0922 | 0.9119 | 0.1008 | 0.9041 | 0.1097 | 0.8961 | 0.1693 |
