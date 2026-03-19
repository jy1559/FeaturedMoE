# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 10.5647 | 0.3686 | 0.1773 | 2.0845 | 0.0000 |
| micro@1 | 11.4329 | 0.2227 | 0.1973 | 2.0127 | 0.7411 |
| mid@1 | 11.6047 | 0.1846 | 0.2614 | 2.2633 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0247 | 0.9756 | 0.0134 | 0.9866 | 0.0118 | 0.9883 |
| micro@1 | 0.0215 | 0.9788 | 0.0121 | 0.9879 | 0.0091 | 0.9910 |
| mid@1 | 0.0112 | 0.9889 | 0.0040 | 0.9960 | 0.0070 | 0.9930 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0325 | 0.9681 | 0.0417 | 0.9591 | 0.0428 | 0.9581 | 0.0374 | 0.9633 | 0.1425 |
| micro@1 | 0.0273 | 0.9730 | 0.0341 | 0.9665 | 0.0318 | 0.9687 | 0.0343 | 0.9662 | 0.1255 |
| mid@1 | 0.0183 | 0.9818 | 0.0185 | 0.9817 | 0.0230 | 0.9773 | 0.0229 | 0.9774 | 0.1100 |
