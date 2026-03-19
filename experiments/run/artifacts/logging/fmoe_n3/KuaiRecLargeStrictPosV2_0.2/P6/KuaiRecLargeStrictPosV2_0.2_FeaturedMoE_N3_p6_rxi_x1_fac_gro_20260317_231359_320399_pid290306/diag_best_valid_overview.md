# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 10.5279 | 0.3739 | 0.2066 | 1.5319 | 0.0000 |
| micro@1 | 8.7169 | 0.6137 | 0.2400 | 1.1549 | 0.4929 |
| mid@1 | 8.2853 | 0.6696 | 0.2460 | 1.4191 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0800 | 0.9231 | 0.0305 | 0.9700 | 0.0935 | 0.9109 |
| micro@1 | 0.0463 | 0.9547 | 0.0190 | 0.9812 | 0.0457 | 0.9554 |
| mid@1 | 0.0636 | 0.9384 | 0.0215 | 0.9787 | 0.0768 | 0.9264 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.1034 | 0.9018 | 0.1193 | 0.8876 | 0.1247 | 0.8828 | 0.1055 | 0.8999 | 0.1395 |
| micro@1 | 0.0719 | 0.9306 | 0.0874 | 0.9163 | 0.0893 | 0.9145 | 0.0797 | 0.9234 | 0.1971 |
| mid@1 | 0.0936 | 0.9107 | 0.0934 | 0.9108 | 0.1088 | 0.8969 | 0.1040 | 0.9013 | 0.1934 |
