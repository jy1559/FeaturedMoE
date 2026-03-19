# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 11.4090 | 0.2276 | 0.1567 | 1.7168 | 0.0000 |
| micro@1 | 10.4770 | 0.3813 | 0.1457 | 1.5163 | 0.6241 |
| mid@1 | 8.8654 | 0.5946 | 0.2091 | 1.5487 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0474 | 0.9537 | 0.0432 | 0.9578 | 0.0053 | 0.9947 |
| micro@1 | 0.0465 | 0.9546 | 0.0416 | 0.9592 | 0.0078 | 0.9922 |
| mid@1 | 0.0337 | 0.9669 | 0.0265 | 0.9739 | 0.0085 | 0.9915 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0603 | 0.9415 | 0.0719 | 0.9307 | 0.0775 | 0.9254 | 0.0650 | 0.9371 | 0.1168 |
| micro@1 | 0.0761 | 0.9268 | 0.0919 | 0.9122 | 0.0942 | 0.9101 | 0.0775 | 0.9254 | 0.1294 |
| mid@1 | 0.0752 | 0.9276 | 0.0688 | 0.9335 | 0.0778 | 0.9252 | 0.0653 | 0.9368 | 0.1661 |
