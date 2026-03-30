# Diagnostic Overview

- split: best_valid
- feature_mode: perturb:zero@both:kw=cat+theme#shift1

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 10.8493 | 0.3257 | 0.4656 | 2.3213 | 0.4964 |
| micro@1 | 10.7998 | 0.3334 | 0.3032 | 2.2399 | 0.7453 |
| mid@1 | 9.3521 | 0.5321 | 0.6070 | 2.1914 | 0.4541 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0125 | 0.9876 | 0.0118 | 0.9882 | 0.0007 | 0.9993 |
| micro@1 | 0.0141 | 0.9860 | 0.0133 | 0.9868 | 0.0009 | 0.9991 |
| mid@1 | 0.0119 | 0.9882 | 0.0103 | 0.9897 | 0.0014 | 0.9986 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0194 | 0.9808 | 0.0280 | 0.9724 | 0.0236 | 0.9766 | 0.0172 | 0.9829 | 0.1018 |
| micro@1 | 0.0168 | 0.9833 | 0.0272 | 0.9732 | 0.0165 | 0.9837 | 0.0207 | 0.9795 | 0.0977 |
| mid@1 | 0.0280 | 0.9724 | 0.0369 | 0.9637 | 0.0200 | 0.9802 | 0.0392 | 0.9615 | 0.1599 |
