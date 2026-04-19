# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 3.9779 | 0.0745 | 0.3048 | 1.2215 | 0.4148 |
| mid@1 | 3.9649 | 0.0941 | 0.3774 | 1.2441 | 0.4365 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0152 | 0.9849 | 0.0152 | 0.9849 | 0.0000 | 1.0000 |
| mid@1 | 0.0196 | 0.9805 | 0.0196 | 0.9805 | 0.0000 | 1.0000 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0216 | 0.9786 | 0.0244 | 0.9759 | 0.0278 | 0.9726 | 0.0220 | 0.9783 | 0.2780 |
| mid@1 | 0.0310 | 0.9694 | 0.0336 | 0.9670 | 0.0358 | 0.9649 | 0.0239 | 0.9764 | 0.2867 |
