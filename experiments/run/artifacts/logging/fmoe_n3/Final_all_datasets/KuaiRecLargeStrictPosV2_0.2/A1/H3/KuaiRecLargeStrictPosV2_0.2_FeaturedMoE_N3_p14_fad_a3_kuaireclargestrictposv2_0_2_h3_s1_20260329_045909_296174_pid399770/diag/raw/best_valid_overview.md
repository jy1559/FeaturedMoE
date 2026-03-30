# Diagnostic Overview

- split: best_valid
- feature_mode: perturb:zero@both:kw=cat+theme#shift1

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 10.9213 | 0.3143 | 0.4672 | 2.3228 | 0.5154 |
| micro@1 | 10.7481 | 0.3413 | 0.2997 | 2.2468 | 0.7530 |
| mid@1 | 9.1605 | 0.5568 | 0.6762 | 2.1930 | 0.3503 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0127 | 0.9873 | 0.0121 | 0.9879 | 0.0006 | 0.9994 |
| micro@1 | 0.0141 | 0.9860 | 0.0132 | 0.9869 | 0.0009 | 0.9991 |
| mid@1 | 0.0116 | 0.9885 | 0.0101 | 0.9899 | 0.0013 | 0.9987 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0196 | 0.9806 | 0.0274 | 0.9729 | 0.0233 | 0.9769 | 0.0175 | 0.9827 | 0.1022 |
| micro@1 | 0.0168 | 0.9833 | 0.0248 | 0.9755 | 0.0165 | 0.9836 | 0.0204 | 0.9798 | 0.0993 |
| mid@1 | 0.0277 | 0.9727 | 0.0359 | 0.9647 | 0.0193 | 0.9809 | 0.0393 | 0.9615 | 0.1712 |
