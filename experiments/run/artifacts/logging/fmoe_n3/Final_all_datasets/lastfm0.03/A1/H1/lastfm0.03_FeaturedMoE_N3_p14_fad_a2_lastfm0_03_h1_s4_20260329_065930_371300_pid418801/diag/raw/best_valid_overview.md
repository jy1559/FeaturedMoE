# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 11.5598 | 0.1951 | 0.2317 | 2.3290 | 0.5077 |
| micro@1 | 11.5548 | 0.1963 | 0.2253 | 2.2861 | 0.6107 |
| mid@1 | 11.5914 | 0.1878 | 0.1949 | 2.0827 | 0.4908 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0173 | 0.9829 | 0.0154 | 0.9847 | 0.0017 | 0.9983 |
| micro@1 | 0.0261 | 0.9742 | 0.0253 | 0.9750 | 0.0009 | 0.9991 |
| mid@1 | 0.0328 | 0.9678 | 0.0273 | 0.9731 | 0.0053 | 0.9947 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0286 | 0.9718 | 0.0280 | 0.9724 | 0.0279 | 0.9725 | 0.0266 | 0.9738 | 0.1038 |
| micro@1 | 0.0419 | 0.9590 | 0.0329 | 0.9676 | 0.0452 | 0.9558 | 0.0346 | 0.9660 | 0.1099 |
| mid@1 | 0.0912 | 0.9128 | 0.0615 | 0.9403 | 0.0889 | 0.9149 | 0.0628 | 0.9392 | 0.1240 |
