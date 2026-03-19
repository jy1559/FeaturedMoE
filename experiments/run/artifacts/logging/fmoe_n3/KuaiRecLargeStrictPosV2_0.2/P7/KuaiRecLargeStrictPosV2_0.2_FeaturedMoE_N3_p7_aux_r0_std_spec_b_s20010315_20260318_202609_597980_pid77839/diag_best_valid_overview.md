# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 4.0946 | 1.3895 | 0.3747 | 0.6852 | 0.0000 |
| micro@1 | 3.9457 | 1.4287 | 0.4571 | 0.6005 | 0.3728 |
| mid@1 | 1.7439 | 2.4251 | 0.7590 | 0.4788 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0620 | 0.9399 | 0.0360 | 0.9646 | 0.0331 | 0.9675 |
| micro@1 | 0.0487 | 0.9524 | 0.0249 | 0.9754 | 0.0394 | 0.9614 |
| mid@1 | 0.0119 | 0.9881 | 0.0066 | 0.9934 | 0.0170 | 0.9832 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0794 | 0.9236 | 0.0875 | 0.9162 | 0.0938 | 0.9105 | 0.0789 | 0.9241 | 0.3509 |
| micro@1 | 0.0828 | 0.9205 | 0.0962 | 0.9083 | 0.0984 | 0.9063 | 0.0742 | 0.9285 | 0.4538 |
| mid@1 | 0.0272 | 0.9732 | 0.0233 | 0.9769 | 0.0199 | 0.9803 | 0.0641 | 0.9379 | 0.7758 |
