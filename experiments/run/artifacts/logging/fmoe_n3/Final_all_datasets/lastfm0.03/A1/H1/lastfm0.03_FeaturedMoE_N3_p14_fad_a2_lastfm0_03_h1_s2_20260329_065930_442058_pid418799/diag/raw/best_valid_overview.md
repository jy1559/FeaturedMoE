# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 11.5527 | 0.1968 | 0.2366 | 2.3312 | 0.5200 |
| micro@1 | 11.5321 | 0.2014 | 0.2198 | 2.2766 | 0.6046 |
| mid@1 | 11.6290 | 0.1786 | 0.2255 | 2.0850 | 0.4937 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0176 | 0.9826 | 0.0150 | 0.9851 | 0.0023 | 0.9977 |
| micro@1 | 0.0250 | 0.9754 | 0.0240 | 0.9763 | 0.0011 | 0.9989 |
| mid@1 | 0.0316 | 0.9689 | 0.0252 | 0.9751 | 0.0064 | 0.9936 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0283 | 0.9721 | 0.0281 | 0.9723 | 0.0274 | 0.9730 | 0.0271 | 0.9732 | 0.1040 |
| micro@1 | 0.0432 | 0.9577 | 0.0317 | 0.9688 | 0.0462 | 0.9549 | 0.0350 | 0.9656 | 0.1117 |
| mid@1 | 0.0938 | 0.9104 | 0.0629 | 0.9390 | 0.0906 | 0.9134 | 0.0635 | 0.9385 | 0.1167 |
