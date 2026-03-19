# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 11.4761 | 0.2137 | 0.2518 | 2.1428 | 0.0000 |
| micro@1 | 9.0516 | 0.5707 | 0.2913 | 1.5174 | 0.4650 |
| mid@1 | 9.8373 | 0.4689 | 0.3641 | 2.0262 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0219 | 0.9783 | 0.0110 | 0.9891 | 0.0110 | 0.9891 |
| micro@1 | 0.0314 | 0.9691 | 0.0135 | 0.9866 | 0.0180 | 0.9822 |
| mid@1 | 0.0179 | 0.9823 | 0.0059 | 0.9941 | 0.0099 | 0.9901 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0299 | 0.9706 | 0.0367 | 0.9640 | 0.0386 | 0.9621 | 0.0311 | 0.9693 | 0.1115 |
| micro@1 | 0.0533 | 0.9481 | 0.0621 | 0.9398 | 0.0636 | 0.9383 | 0.0583 | 0.9434 | 0.1961 |
| mid@1 | 0.0298 | 0.9707 | 0.0290 | 0.9714 | 0.0354 | 0.9652 | 0.0368 | 0.9639 | 0.1891 |
