# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 10.7679 | 0.3383 | 0.3263 | 2.1564 | 0.6152 |
| micro@1 | 7.9952 | 0.7077 | 0.2796 | 1.7430 | 0.7550 |
| mid@1 | 9.5874 | 0.5016 | 0.4621 | 2.0578 | 0.5727 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0206 | 0.9796 | 0.0183 | 0.9819 | 0.0027 | 0.9973 |
| micro@1 | 0.0447 | 0.9563 | 0.0377 | 0.9630 | 0.0161 | 0.9842 |
| mid@1 | 0.0226 | 0.9776 | 0.0152 | 0.9850 | 0.0079 | 0.9921 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0353 | 0.9653 | 0.0424 | 0.9585 | 0.0428 | 0.9581 | 0.0378 | 0.9629 | 0.1324 |
| micro@1 | 0.0704 | 0.9320 | 0.0680 | 0.9343 | 0.0761 | 0.9267 | 0.0474 | 0.9537 | 0.1891 |
| mid@1 | 0.0585 | 0.9432 | 0.0502 | 0.9511 | 0.0643 | 0.9377 | 0.0416 | 0.9593 | 0.1894 |
