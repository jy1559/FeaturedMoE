# Diagnostic Overview

- split: best_valid
- feature_mode: perturb:zero@both:kw=cat+theme#shift1

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 7.2613 | 0.8078 | 0.5310 | 1.9956 | 0.3118 |
| micro@1 | 3.7926 | 1.4711 | 0.8651 | 1.3927 | 0.1403 |
| mid@1 | 8.7267 | 0.6124 | 0.4400 | 2.1725 | 0.3520 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0161 | 0.9841 | 0.0062 | 0.9938 | 0.0076 | 0.9924 |
| micro@1 | 0.0074 | 0.9926 | 0.0059 | 0.9941 | 0.0014 | 0.9986 |
| mid@1 | 0.0081 | 0.9919 | 0.0046 | 0.9954 | 0.0033 | 0.9967 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0337 | 0.9668 | 0.0414 | 0.9595 | 0.0320 | 0.9685 | 0.0368 | 0.9638 | 0.1567 |
| micro@1 | 0.0135 | 0.9866 | 0.0275 | 0.9729 | 0.0131 | 0.9870 | 0.0092 | 0.9909 | 0.2907 |
| mid@1 | 0.0260 | 0.9744 | 0.0258 | 0.9745 | 0.0292 | 0.9712 | 0.0162 | 0.9839 | 0.1353 |
