# Diagnostic Overview

- split: best_valid
- feature_mode: runtime:unknown|ablation:none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 7.9837 | 0.0452 | 0.1911 | 1.9872 | 0.7544 |
| mid@1 | 7.9642 | 0.0670 | 0.2050 | 1.9089 | 0.6463 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0109 | 0.9892 | 0.0050 | 0.9950 | 0.0058 | 0.9942 |
| mid@1 | 0.0172 | 0.9829 | 0.0075 | 0.9926 | 0.0098 | 0.9903 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0136 | 0.9865 | 0.0129 | 0.9872 | 0.0134 | 0.9867 | 0.0127 | 0.9874 | 0.1356 |
| mid@1 | 0.0448 | 0.9562 | 0.0209 | 0.9793 | 0.0420 | 0.9589 | 0.0368 | 0.9639 | 0.1446 |
