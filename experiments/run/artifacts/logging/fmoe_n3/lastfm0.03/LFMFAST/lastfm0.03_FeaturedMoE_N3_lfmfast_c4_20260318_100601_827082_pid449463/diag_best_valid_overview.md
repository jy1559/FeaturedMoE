# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 11.4912 | 0.2104 | 0.1188 | 1.8628 | 0.0000 |
| micro@1 | 11.5680 | 0.1933 | 0.1516 | 1.9723 | 0.5727 |
| mid@1 | 11.4411 | 0.2210 | 0.1339 | 1.8600 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0413 | 0.9596 | 0.0181 | 0.9821 | 0.0268 | 0.9736 |
| micro@1 | 0.0199 | 0.9803 | 0.0064 | 0.9936 | 0.0138 | 0.9863 |
| mid@1 | 0.0315 | 0.9690 | 0.0134 | 0.9867 | 0.0202 | 0.9800 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0731 | 0.9295 | 0.0639 | 0.9381 | 0.0692 | 0.9331 | 0.0638 | 0.9382 | 0.1138 |
| micro@1 | 0.0714 | 0.9311 | 0.0629 | 0.9391 | 0.0978 | 0.9069 | 0.0555 | 0.9460 | 0.1135 |
| mid@1 | 0.0848 | 0.9187 | 0.0528 | 0.9486 | 0.0952 | 0.9092 | 0.0694 | 0.9329 | 0.1116 |
