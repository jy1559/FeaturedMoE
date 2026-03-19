# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 3.7215 | 1.4915 | 0.6971 | 1.1465 | 0.0000 |
| micro@1 | 3.3012 | 1.6233 | 0.4487 | 0.6042 | 0.4093 |
| mid@1 | 2.1542 | 2.1379 | 0.7262 | 0.8269 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0406 | 0.9602 | 0.0169 | 0.9832 | 0.0239 | 0.9764 |
| micro@1 | 0.0480 | 0.9532 | 0.0283 | 0.9721 | 0.0311 | 0.9695 |
| mid@1 | 0.0202 | 0.9800 | 0.0059 | 0.9941 | 0.0129 | 0.9872 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0591 | 0.9426 | 0.0676 | 0.9346 | 0.0727 | 0.9299 | 0.0627 | 0.9392 | 0.4798 |
| micro@1 | 0.0809 | 0.9223 | 0.0993 | 0.9055 | 0.1024 | 0.9026 | 0.0833 | 0.9201 | 0.4262 |
| mid@1 | 0.0374 | 0.9633 | 0.0316 | 0.9689 | 0.0324 | 0.9681 | 0.0570 | 0.9446 | 0.6927 |
