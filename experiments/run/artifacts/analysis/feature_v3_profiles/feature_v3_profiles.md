# Feature V3 Profiles

## Structure

| dataset | rows | sessions | users | items | avg_sess | p90_sess | eff@10 | top100 share |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| KuaiRecLargeStrictPosV2_0.2 | 287,411 | 24,458 | 1,122 | 6,477 | 11.75 | 24.0 | 188,726 | 0.117 |
| lastfm0.03 | 470,408 | 25,089 | 130 | 52,510 | 18.75 | 44.0 | 223,542 | 0.026 |

## Macro Context Availability

| dataset | mac5 mean | mac5 p90 | mac10 mean | mac10 p90 |
| --- | ---: | ---: | ---: | ---: |
| KuaiRecLargeStrictPosV2_0.2 | 0.908 | 1.000 | 0.824 | 1.000 |
| lastfm0.03 | 0.985 | 1.000 | 0.973 | 1.000 |

## Light V2 Overlap

### KuaiRecLargeStrictPosV2_0.2

| feature | v2 mean | v3 mean | v2 p90 | v3 p90 |
| --- | ---: | ---: | ---: | ---: |
| mid_valid_r | 0.582 | 0.847 | 1.000 | 1.000 |
| mid_int_std | -0.085 | 0.666 | 1.090 | 0.830 |
| mid_novel_r | 0.803 | 0.875 | 1.000 | 1.000 |
| mic_valid_r | 0.830 | 0.745 | 1.000 | 1.000 |
| mic_is_recons | 0.098 | 0.098 | 0.000 | 0.000 |

### lastfm0.03

| feature | v2 mean | v3 mean | v2 p90 | v3 p90 |
| --- | ---: | ---: | ---: | ---: |
| mid_valid_r | 0.721 | 0.949 | 1.000 | 1.000 |
| mid_int_std | 0.043 | 0.626 | 1.559 | 0.745 |
| mid_novel_r | 0.257 | 0.274 | 0.900 | 0.800 |
| mic_valid_r | 0.893 | 0.840 | 1.000 | 1.000 |
| mic_is_recons | 0.102 | 0.102 | 1.000 | 1.000 |

