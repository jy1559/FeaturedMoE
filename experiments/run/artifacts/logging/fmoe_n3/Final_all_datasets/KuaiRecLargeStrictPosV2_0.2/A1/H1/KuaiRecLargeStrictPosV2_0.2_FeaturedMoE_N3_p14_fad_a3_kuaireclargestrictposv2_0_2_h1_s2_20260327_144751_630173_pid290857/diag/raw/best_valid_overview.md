# Diagnostic Overview

- split: best_valid
- feature_mode: perturb:zero@both:kw=cat+theme#shift1

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 9.1312 | 0.5605 | 0.4234 | 2.0111 | 0.4756 |
| micro@1 | 9.0758 | 0.5676 | 0.4907 | 1.9426 | 0.4736 |
| mid@1 | 8.6245 | 0.6256 | 0.4537 | 2.0974 | 0.4912 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0276 | 0.9728 | 0.0135 | 0.9866 | 0.0132 | 0.9869 |
| micro@1 | 0.0244 | 0.9759 | 0.0191 | 0.9811 | 0.0052 | 0.9949 |
| mid@1 | 0.0130 | 0.9870 | 0.0079 | 0.9921 | 0.0072 | 0.9928 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0511 | 0.9502 | 0.0720 | 0.9306 | 0.0587 | 0.9430 | 0.0743 | 0.9284 | 0.1339 |
| micro@1 | 0.0535 | 0.9479 | 0.0538 | 0.9476 | 0.0560 | 0.9455 | 0.0300 | 0.9704 | 0.1411 |
| mid@1 | 0.0334 | 0.9671 | 0.0419 | 0.9590 | 0.0483 | 0.9528 | 0.0292 | 0.9712 | 0.1243 |
