# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 11.2965 | 0.2495 | 0.2493 | 1.9005 | 0.0000 |
| micro@1 | 11.3103 | 0.2469 | 0.1599 | 1.7315 | 0.5273 |
| mid@1 | 9.7304 | 0.4830 | 0.3084 | 1.7182 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0497 | 0.9515 | 0.0236 | 0.9767 | 0.0313 | 0.9692 |
| micro@1 | 0.0403 | 0.9605 | 0.0183 | 0.9819 | 0.0276 | 0.9728 |
| mid@1 | 0.0419 | 0.9589 | 0.0207 | 0.9795 | 0.0227 | 0.9776 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0617 | 0.9401 | 0.0668 | 0.9353 | 0.0728 | 0.9298 | 0.0644 | 0.9376 | 0.1370 |
| micro@1 | 0.0697 | 0.9327 | 0.0816 | 0.9217 | 0.0874 | 0.9163 | 0.0642 | 0.9378 | 0.1201 |
| mid@1 | 0.0627 | 0.9392 | 0.0608 | 0.9410 | 0.0734 | 0.9292 | 0.0704 | 0.9320 | 0.1836 |
