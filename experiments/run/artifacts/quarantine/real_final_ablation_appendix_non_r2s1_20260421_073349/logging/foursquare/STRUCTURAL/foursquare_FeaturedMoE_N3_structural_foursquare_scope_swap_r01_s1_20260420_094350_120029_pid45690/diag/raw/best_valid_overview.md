# Diagnostic Overview

- split: best_valid
- feature_mode: runtime:unknown|ablation:none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 15.7500 | 0.1260 | 0.1105 | 2.5394 | 0.7360 |
| mid@1 | 15.7801 | 0.1181 | 0.1178 | 2.4721 | 0.6855 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0582 | 0.9435 | 0.0145 | 0.9856 | 0.0439 | 0.9571 |
| mid@1 | 0.0644 | 0.9376 | 0.0150 | 0.9851 | 0.0494 | 0.9518 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0680 | 0.9342 | 0.0672 | 0.9350 | 0.0660 | 0.9362 | 0.0634 | 0.9386 | 0.0812 |
| mid@1 | 0.0819 | 0.9213 | 0.0813 | 0.9219 | 0.0769 | 0.9260 | 0.0755 | 0.9273 | 0.0823 |
