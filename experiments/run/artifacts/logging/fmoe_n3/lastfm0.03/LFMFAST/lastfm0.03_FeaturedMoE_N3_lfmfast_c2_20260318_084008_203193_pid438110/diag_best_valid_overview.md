# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 10.6192 | 0.3606 | 0.1263 | 1.4596 | 0.0000 |
| micro@1 | 10.9973 | 0.3020 | 0.1241 | 1.4496 | 0.5385 |
| mid@1 | 11.1136 | 0.2824 | 0.1256 | 1.3925 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.1001 | 0.9048 | 0.0451 | 0.9559 | 0.0904 | 0.9137 |
| micro@1 | 0.0641 | 0.9379 | 0.0214 | 0.9789 | 0.0558 | 0.9458 |
| mid@1 | 0.0903 | 0.9136 | 0.0423 | 0.9585 | 0.0866 | 0.9171 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.1400 | 0.8694 | 0.1344 | 0.8742 | 0.1367 | 0.8722 | 0.1282 | 0.8797 | 0.1356 |
| micro@1 | 0.1634 | 0.8492 | 0.1762 | 0.8384 | 0.2122 | 0.8088 | 0.1266 | 0.8811 | 0.1308 |
| mid@1 | 0.1470 | 0.8633 | 0.1386 | 0.8706 | 0.1652 | 0.8477 | 0.1354 | 0.8734 | 0.1143 |
