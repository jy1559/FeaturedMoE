# Diagnostic Overview

- split: best_valid
- feature_mode: runtime:unknown|ablation:none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 15.8273 | 0.1045 | 0.0949 | 2.4708 | 0.7276 |
| mid@1 | 15.7439 | 0.1275 | 0.1330 | 2.3703 | 0.7149 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0742 | 0.9285 | 0.0197 | 0.9805 | 0.0548 | 0.9467 |
| mid@1 | 0.0797 | 0.9234 | 0.0207 | 0.9795 | 0.0596 | 0.9422 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0866 | 0.9170 | 0.0846 | 0.9189 | 0.0834 | 0.9200 | 0.0794 | 0.9237 | 0.0787 |
| mid@1 | 0.1048 | 0.9005 | 0.1059 | 0.8996 | 0.0999 | 0.9049 | 0.0937 | 0.9106 | 0.0892 |
