# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 11.4837 | 0.2120 | 0.1969 | 1.9285 | 0.0000 |
| micro@1 | 9.4043 | 0.5254 | 0.2222 | 1.1719 | 0.5016 |
| mid@1 | 11.2949 | 0.2499 | 0.2721 | 2.0029 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0509 | 0.9504 | 0.0303 | 0.9701 | 0.0222 | 0.9781 |
| micro@1 | 0.0590 | 0.9427 | 0.0364 | 0.9643 | 0.0259 | 0.9744 |
| mid@1 | 0.0326 | 0.9679 | 0.0161 | 0.9841 | 0.0181 | 0.9820 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0623 | 0.9396 | 0.0690 | 0.9333 | 0.0739 | 0.9288 | 0.0649 | 0.9371 | 0.1135 |
| micro@1 | 0.0979 | 0.9067 | 0.1154 | 0.8910 | 0.1196 | 0.8872 | 0.0949 | 0.9094 | 0.1928 |
| mid@1 | 0.0487 | 0.9525 | 0.0484 | 0.9527 | 0.0560 | 0.9455 | 0.0539 | 0.9475 | 0.1250 |
