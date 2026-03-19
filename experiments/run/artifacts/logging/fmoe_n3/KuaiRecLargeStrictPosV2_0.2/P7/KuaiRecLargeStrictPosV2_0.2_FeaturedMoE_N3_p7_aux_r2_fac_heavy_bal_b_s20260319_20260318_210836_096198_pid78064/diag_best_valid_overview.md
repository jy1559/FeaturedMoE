# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 11.3942 | 0.2306 | 0.1462 | 1.6033 | 0.0000 |
| micro@1 | 10.2880 | 0.4079 | 0.1573 | 1.4045 | 0.6016 |
| mid@1 | 9.7756 | 0.4770 | 0.2013 | 1.6374 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0507 | 0.9505 | 0.0454 | 0.9556 | 0.0066 | 0.9934 |
| micro@1 | 0.0504 | 0.9508 | 0.0440 | 0.9570 | 0.0108 | 0.9893 |
| mid@1 | 0.0383 | 0.9624 | 0.0296 | 0.9708 | 0.0103 | 0.9898 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0652 | 0.9369 | 0.0794 | 0.9237 | 0.0851 | 0.9184 | 0.0712 | 0.9313 | 0.1143 |
| micro@1 | 0.0808 | 0.9224 | 0.0983 | 0.9064 | 0.1026 | 0.9025 | 0.0861 | 0.9175 | 0.1412 |
| mid@1 | 0.0826 | 0.9207 | 0.0777 | 0.9253 | 0.0879 | 0.9158 | 0.0709 | 0.9315 | 0.1523 |
