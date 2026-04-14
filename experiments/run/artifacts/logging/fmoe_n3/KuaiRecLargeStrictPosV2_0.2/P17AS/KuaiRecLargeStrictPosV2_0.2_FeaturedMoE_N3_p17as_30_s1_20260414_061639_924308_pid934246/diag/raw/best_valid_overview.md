# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 15.5382 | 0.1724 | 0.7420 | 2.7544 | 0.0000 |
| micro@1 | 15.8787 | 0.0874 | 0.5583 | 2.7616 | 0.5410 |
| mid@1 | 15.9227 | 0.0697 | 0.4900 | 2.7682 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0003 | 0.9997 | 0.0003 | 0.9997 | 0.0000 | 1.0000 |
| micro@1 | 0.0001 | 0.9999 | 0.0001 | 0.9999 | 0.0000 | 1.0000 |
| mid@1 | 0.0002 | 0.9998 | 0.0002 | 0.9998 | 0.0000 | 1.0000 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0005 | 0.9995 | 0.0009 | 0.9991 | 0.0009 | 0.9991 | 0.0009 | 0.9991 | 0.0749 |
| micro@1 | 0.0003 | 0.9997 | 0.0002 | 0.9998 | 0.0004 | 0.9996 | 0.0004 | 0.9996 | 0.0698 |
| mid@1 | 0.0007 | 0.9993 | 0.0008 | 0.9992 | 0.0007 | 0.9993 | 0.0007 | 0.9993 | 0.0664 |
