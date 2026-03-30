# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 11.4962 | 0.2093 | 0.4271 | 2.2883 | 0.4604 |
| micro@1 | 11.3266 | 0.2438 | 0.3618 | 2.2850 | 0.4989 |
| mid@1 | 11.6103 | 0.1832 | 0.1965 | 2.2542 | 0.4078 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0173 | 0.9828 | 0.0149 | 0.9852 | 0.0028 | 0.9972 |
| micro@1 | 0.0177 | 0.9825 | 0.0170 | 0.9831 | 0.0008 | 0.9992 |
| mid@1 | 0.0185 | 0.9817 | 0.0095 | 0.9905 | 0.0093 | 0.9908 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0291 | 0.9713 | 0.0305 | 0.9700 | 0.0315 | 0.9690 | 0.0290 | 0.9714 | 0.1247 |
| micro@1 | 0.0478 | 0.9534 | 0.0335 | 0.9670 | 0.0488 | 0.9524 | 0.0302 | 0.9703 | 0.1185 |
| mid@1 | 0.0764 | 0.9264 | 0.0438 | 0.9572 | 0.0684 | 0.9339 | 0.0579 | 0.9437 | 0.1080 |
