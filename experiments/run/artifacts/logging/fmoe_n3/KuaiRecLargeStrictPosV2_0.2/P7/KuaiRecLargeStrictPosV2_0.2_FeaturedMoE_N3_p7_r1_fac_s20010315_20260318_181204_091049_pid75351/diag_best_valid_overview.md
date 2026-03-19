# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 11.4393 | 0.2214 | 0.2620 | 2.0800 | 0.0000 |
| micro@1 | 9.8883 | 0.4621 | 0.2671 | 1.5873 | 0.5136 |
| mid@1 | 9.3408 | 0.5336 | 0.4079 | 1.9498 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0248 | 0.9755 | 0.0128 | 0.9873 | 0.0128 | 0.9873 |
| micro@1 | 0.0325 | 0.9680 | 0.0142 | 0.9859 | 0.0201 | 0.9802 |
| mid@1 | 0.0198 | 0.9804 | 0.0069 | 0.9931 | 0.0111 | 0.9890 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0339 | 0.9666 | 0.0417 | 0.9592 | 0.0433 | 0.9576 | 0.0357 | 0.9649 | 0.1174 |
| micro@1 | 0.0553 | 0.9462 | 0.0642 | 0.9378 | 0.0647 | 0.9373 | 0.0602 | 0.9416 | 0.1776 |
| mid@1 | 0.0332 | 0.9674 | 0.0321 | 0.9684 | 0.0385 | 0.9622 | 0.0413 | 0.9595 | 0.2095 |
