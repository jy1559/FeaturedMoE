# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 11.2903 | 0.2507 | 0.2038 | 1.8468 | 0.0000 |
| micro@1 | 11.3263 | 0.2439 | 0.1807 | 1.9201 | 0.4981 |
| mid@1 | 9.0324 | 0.5732 | 0.3303 | 1.5934 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0516 | 0.9497 | 0.0248 | 0.9755 | 0.0330 | 0.9675 |
| micro@1 | 0.0335 | 0.9670 | 0.0139 | 0.9861 | 0.0220 | 0.9782 |
| mid@1 | 0.0458 | 0.9552 | 0.0250 | 0.9753 | 0.0236 | 0.9767 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0640 | 0.9380 | 0.0700 | 0.9324 | 0.0759 | 0.9269 | 0.0672 | 0.9350 | 0.1279 |
| micro@1 | 0.0597 | 0.9421 | 0.0700 | 0.9324 | 0.0742 | 0.9284 | 0.0520 | 0.9493 | 0.1224 |
| mid@1 | 0.0693 | 0.9331 | 0.0660 | 0.9361 | 0.0782 | 0.9248 | 0.0798 | 0.9233 | 0.2115 |
