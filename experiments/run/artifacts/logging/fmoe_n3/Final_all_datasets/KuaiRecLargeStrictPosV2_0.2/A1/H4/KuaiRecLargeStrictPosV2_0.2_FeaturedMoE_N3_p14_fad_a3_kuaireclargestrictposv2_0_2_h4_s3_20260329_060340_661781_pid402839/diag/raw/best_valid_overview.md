# Diagnostic Overview

- split: best_valid
- feature_mode: perturb:zero@both:kw=cat+theme#shift1

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 10.3068 | 0.4053 | 0.6152 | 2.3099 | 0.6280 |
| micro@1 | 11.0201 | 0.2982 | 0.6442 | 2.3090 | 0.5069 |
| mid@1 | 7.8309 | 0.7297 | 0.7868 | 2.1367 | 0.1197 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0068 | 0.9932 | 0.0063 | 0.9937 | 0.0004 | 0.9996 |
| micro@1 | 0.0099 | 0.9901 | 0.0097 | 0.9903 | 0.0001 | 0.9999 |
| mid@1 | 0.0133 | 0.9868 | 0.0122 | 0.9879 | 0.0009 | 0.9991 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0108 | 0.9893 | 0.0135 | 0.9866 | 0.0120 | 0.9880 | 0.0090 | 0.9910 | 0.1241 |
| micro@1 | 0.0120 | 0.9880 | 0.0174 | 0.9827 | 0.0112 | 0.9889 | 0.0165 | 0.9836 | 0.1068 |
| mid@1 | 0.0304 | 0.9700 | 0.0378 | 0.9629 | 0.0197 | 0.9805 | 0.0418 | 0.9591 | 0.2092 |
