# Diagnostic Overview

- split: best_valid
- feature_mode: runtime:unknown|ablation:none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 15.7731 | 0.1199 | 0.1132 | 2.5808 | 0.7427 |
| mid@1 | 15.8278 | 0.1043 | 0.1110 | 2.5661 | 0.7208 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0445 | 0.9565 | 0.0085 | 0.9915 | 0.0358 | 0.9648 |
| mid@1 | 0.0446 | 0.9564 | 0.0110 | 0.9890 | 0.0340 | 0.9665 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0541 | 0.9473 | 0.0539 | 0.9476 | 0.0526 | 0.9487 | 0.0488 | 0.9524 | 0.0760 |
| mid@1 | 0.0560 | 0.9455 | 0.0552 | 0.9463 | 0.0560 | 0.9455 | 0.0505 | 0.9507 | 0.0786 |
