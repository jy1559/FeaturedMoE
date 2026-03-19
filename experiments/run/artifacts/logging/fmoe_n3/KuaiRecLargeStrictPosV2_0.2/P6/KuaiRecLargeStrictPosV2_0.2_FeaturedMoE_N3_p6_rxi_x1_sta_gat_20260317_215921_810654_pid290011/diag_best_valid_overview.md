# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 10.5093 | 0.3766 | 0.2649 | 1.4644 | 0.0000 |
| micro@1 | 10.3098 | 0.4049 | 0.1830 | 1.1737 | 0.4954 |
| mid@1 | 8.1794 | 0.6834 | 0.3013 | 1.2099 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0962 | 0.9083 | 0.0445 | 0.9565 | 0.0970 | 0.9076 |
| micro@1 | 0.0576 | 0.9441 | 0.0326 | 0.9679 | 0.0671 | 0.9351 |
| mid@1 | 0.0482 | 0.9529 | 0.0207 | 0.9795 | 0.0686 | 0.9338 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.1143 | 0.8920 | 0.1255 | 0.8820 | 0.1350 | 0.8738 | 0.1217 | 0.8854 | 0.1615 |
| micro@1 | 0.0956 | 0.9089 | 0.1164 | 0.8901 | 0.1260 | 0.8816 | 0.0906 | 0.9134 | 0.1535 |
| mid@1 | 0.0740 | 0.9287 | 0.0688 | 0.9336 | 0.0815 | 0.9217 | 0.0985 | 0.9062 | 0.2271 |
