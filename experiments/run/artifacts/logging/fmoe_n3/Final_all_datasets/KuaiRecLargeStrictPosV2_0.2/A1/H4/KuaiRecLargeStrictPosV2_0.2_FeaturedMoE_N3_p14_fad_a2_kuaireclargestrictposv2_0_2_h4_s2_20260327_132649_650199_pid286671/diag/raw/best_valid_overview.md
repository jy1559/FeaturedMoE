# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 4.6606 | 1.2549 | 0.6959 | 1.6303 | 0.2954 |
| micro@1 | 2.6149 | 1.8945 | 0.9989 | 1.0993 | 0.0011 |
| mid@1 | 8.6705 | 0.6197 | 0.6721 | 2.1798 | 0.3212 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0126 | 0.9875 | 0.0096 | 0.9904 | 0.0027 | 0.9973 |
| micro@1 | 0.0018 | 0.9982 | 0.0015 | 0.9985 | 0.0034 | 0.9966 |
| mid@1 | 0.0110 | 0.9890 | 0.0031 | 0.9969 | 0.0081 | 0.9919 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0221 | 0.9782 | 0.0314 | 0.9691 | 0.0313 | 0.9692 | 0.0250 | 0.9753 | 0.2981 |
| micro@1 | 0.0044 | 0.9956 | 0.0057 | 0.9943 | 0.0039 | 0.9962 | 0.0027 | 0.9973 | 0.5381 |
| mid@1 | 0.0284 | 0.9720 | 0.0345 | 0.9661 | 0.0345 | 0.9660 | 0.0289 | 0.9715 | 0.2149 |
