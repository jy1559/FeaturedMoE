# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 5.8959 | 0.1329 | 0.3314 | 1.3546 | 0.0000 |
| micro@1 | 4.7385 | 0.5160 | 0.3393 | 1.1588 | 0.5314 |
| mid@1 | 5.1681 | 0.4012 | 0.3361 | 1.2396 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0279 | 0.9724 | 0.0068 | 0.9932 | 0.0225 | 0.9777 |
| micro@1 | 0.0204 | 0.9798 | 0.0036 | 0.9964 | 0.0149 | 0.9853 |
| mid@1 | 0.0269 | 0.9734 | 0.0111 | 0.9889 | 0.0129 | 0.9872 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0340 | 0.9666 | 0.0463 | 0.9548 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1038 |
| micro@1 | 0.0318 | 0.9687 | 0.0375 | 0.9632 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1585 |
| mid@1 | 0.0559 | 0.9456 | 0.0262 | 0.9742 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1396 |
