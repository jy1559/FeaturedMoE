# Diagnostic Overview

- split: best_valid
- feature_mode: runtime:unknown|ablation:none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 9.2948 | 0.5395 | 0.8476 | 2.3052 | 0.0000 |
| micro@1 | 7.7459 | 0.7411 | 0.7030 | 2.0340 | 0.2962 |
| mid@1 | 8.1916 | 0.6818 | 0.8392 | 2.1117 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0055 | 0.9945 | 0.0013 | 0.9987 | 0.0040 | 0.9960 |
| micro@1 | 0.0039 | 0.9961 | 0.0010 | 0.9990 | 0.0039 | 0.9962 |
| mid@1 | 0.0080 | 0.9921 | 0.0008 | 0.9992 | 0.0070 | 0.9930 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0119 | 0.9882 | 0.0149 | 0.9852 | 0.0166 | 0.9835 | 0.0108 | 0.9892 | 0.1890 |
| micro@1 | 0.0107 | 0.9894 | 0.0222 | 0.9780 | 0.0149 | 0.9852 | 0.0113 | 0.9888 | 0.2644 |
| mid@1 | 0.0298 | 0.9706 | 0.0264 | 0.9740 | 0.0321 | 0.9684 | 0.0261 | 0.9742 | 0.2047 |
