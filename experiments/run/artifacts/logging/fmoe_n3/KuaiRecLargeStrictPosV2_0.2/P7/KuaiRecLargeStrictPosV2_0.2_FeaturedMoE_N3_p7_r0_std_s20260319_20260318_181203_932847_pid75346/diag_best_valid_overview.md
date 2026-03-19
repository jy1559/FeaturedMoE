# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 11.0874 | 0.2869 | 0.2015 | 1.8671 | 0.0000 |
| micro@1 | 11.1147 | 0.2822 | 0.2005 | 1.7557 | 0.4563 |
| mid@1 | 9.1719 | 0.5553 | 0.3018 | 1.7122 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0474 | 0.9537 | 0.0232 | 0.9771 | 0.0325 | 0.9681 |
| micro@1 | 0.0390 | 0.9618 | 0.0196 | 0.9806 | 0.0256 | 0.9747 |
| mid@1 | 0.0455 | 0.9555 | 0.0237 | 0.9766 | 0.0222 | 0.9780 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0602 | 0.9416 | 0.0652 | 0.9368 | 0.0707 | 0.9317 | 0.0621 | 0.9398 | 0.1270 |
| micro@1 | 0.0673 | 0.9349 | 0.0766 | 0.9262 | 0.0828 | 0.9206 | 0.0588 | 0.9429 | 0.1317 |
| mid@1 | 0.0682 | 0.9341 | 0.0671 | 0.9351 | 0.0814 | 0.9218 | 0.0775 | 0.9254 | 0.2020 |
