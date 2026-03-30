# Diagnostic Overview

- split: best_valid
- feature_mode: perturb:zero@both:kw=cat+theme#shift1

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 10.1913 | 0.4213 | 0.6708 | 2.3063 | 0.5187 |
| micro@1 | 10.6691 | 0.3532 | 0.6844 | 2.2881 | 0.4800 |
| mid@1 | 8.4860 | 0.6435 | 0.8085 | 2.2154 | 0.0706 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0069 | 0.9931 | 0.0064 | 0.9937 | 0.0004 | 0.9996 |
| micro@1 | 0.0112 | 0.9888 | 0.0111 | 0.9890 | 0.0001 | 0.9999 |
| mid@1 | 0.0102 | 0.9898 | 0.0097 | 0.9904 | 0.0004 | 0.9996 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0104 | 0.9897 | 0.0099 | 0.9902 | 0.0099 | 0.9902 | 0.0085 | 0.9916 | 0.1241 |
| micro@1 | 0.0140 | 0.9861 | 0.0202 | 0.9800 | 0.0127 | 0.9874 | 0.0186 | 0.9816 | 0.1159 |
| mid@1 | 0.0196 | 0.9806 | 0.0247 | 0.9756 | 0.0114 | 0.9886 | 0.0334 | 0.9672 | 0.1981 |
