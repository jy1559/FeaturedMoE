# Diagnostic Overview

- split: best_valid
- feature_mode: perturb:zero@both:kw=cat+theme#shift1

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 10.4252 | 0.3887 | 0.6099 | 2.3213 | 0.6309 |
| micro@1 | 11.0659 | 0.2905 | 0.6349 | 2.3044 | 0.5127 |
| mid@1 | 7.6035 | 0.7604 | 0.7912 | 2.1144 | 0.1098 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0065 | 0.9935 | 0.0060 | 0.9941 | 0.0004 | 0.9996 |
| micro@1 | 0.0104 | 0.9896 | 0.0102 | 0.9898 | 0.0002 | 0.9998 |
| mid@1 | 0.0129 | 0.9872 | 0.0116 | 0.9885 | 0.0011 | 0.9989 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0101 | 0.9899 | 0.0124 | 0.9877 | 0.0113 | 0.9888 | 0.0085 | 0.9916 | 0.1221 |
| micro@1 | 0.0127 | 0.9874 | 0.0194 | 0.9808 | 0.0119 | 0.9882 | 0.0176 | 0.9825 | 0.1059 |
| mid@1 | 0.0322 | 0.9683 | 0.0407 | 0.9601 | 0.0205 | 0.9797 | 0.0440 | 0.9569 | 0.2146 |
