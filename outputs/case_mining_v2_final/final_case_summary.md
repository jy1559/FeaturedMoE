# Final Case Eval Summary

Permissive cap: 128 sessions per group, per dataset, per split.

## KuaiRecLargeStrictPosV2_0.2

| dataset                     | split   | tier       |   selected_entries |   selected_unique_sessions |   full_sessions |   session_pct_of_split |   selection_score_mean |   selection_score_median |   selection_score_min |   selection_score_max |   core_score_mean |   balance_score_mean |   effective_quota_after_cap |
|:----------------------------|:--------|:-----------|-------------------:|---------------------------:|----------------:|-----------------------:|-----------------------:|-------------------------:|----------------------:|----------------------:|------------------:|---------------------:|----------------------------:|
| KuaiRecLargeStrictPosV2_0.2 | test    | permissive |                904 |                        709 |            3669 |                 0.1932 |                 0.6777 |                   0.6789 |                0.4735 |                0.9932 |            0.7136 |               0.4847 |                         113 |
| KuaiRecLargeStrictPosV2_0.2 | test    | pure       |                640 |                        594 |            3669 |                 0.1619 |                 0.7907 |                   0.8280 |                0.5496 |                0.9944 |            0.8433 |               0.5027 |                          80 |
| KuaiRecLargeStrictPosV2_0.2 | valid   | permissive |                496 |                        449 |            3669 |                 0.1224 |                 0.6120 |                   0.6000 |                0.1660 |                0.9907 |            0.6652 |               0.3150 |                          62 |
| KuaiRecLargeStrictPosV2_0.2 | valid   | pure       |                752 |                        655 |            3669 |                 0.1785 |                 0.7253 |                   0.7410 |                0.1825 |                0.9944 |            0.8064 |               0.2894 |                          94 |

Tier dataset sizes:

| dataset                     | split   |   subset_rows |   subset_sessions | tier       |   full_rows |   full_sessions |   row_pct_of_split |   session_pct_of_split |
|:----------------------------|:--------|--------------:|------------------:|:-----------|------------:|----------------:|-------------------:|-----------------------:|
| KuaiRecLargeStrictPosV2_0.2 | valid   |          8666 |               655 | pure       |       33853 |            3669 |             0.2560 |                 0.1785 |
| KuaiRecLargeStrictPosV2_0.2 | test    |          3967 |               594 | pure       |       14614 |            3669 |             0.2715 |                 0.1619 |
| KuaiRecLargeStrictPosV2_0.2 | valid   |          5858 |               449 | permissive |       33853 |            3669 |             0.1730 |                 0.1224 |
| KuaiRecLargeStrictPosV2_0.2 | test    |          4657 |               709 | permissive |       14614 |            3669 |             0.3187 |                 0.1932 |

All-tier union dataset size:

| dataset                     | split   |   subset_rows |   subset_sessions |   full_rows |   full_sessions |   row_pct_of_split |   session_pct_of_split |
|:----------------------------|:--------|--------------:|------------------:|------------:|----------------:|-------------------:|-----------------------:|
| KuaiRecLargeStrictPosV2_0.2 | valid   |         12318 |               979 |       33853 |            3669 |             0.3639 |                 0.2668 |
| KuaiRecLargeStrictPosV2_0.2 | test    |          6956 |              1133 |       14614 |            3669 |             0.4760 |                 0.3088 |

## lastfm0.03

| dataset    | split   | tier       |   selected_entries |   selected_unique_sessions |   full_sessions |   session_pct_of_split |   selection_score_mean |   selection_score_median |   selection_score_min |   selection_score_max |   core_score_mean |   balance_score_mean |   effective_quota_after_cap |
|:-----------|:--------|:-----------|-------------------:|---------------------------:|----------------:|-----------------------:|-----------------------:|-------------------------:|----------------------:|----------------------:|------------------:|---------------------:|----------------------------:|
| lastfm0.03 | test    | permissive |                104 |                         98 |            3764 |                 0.0260 |                 0.7236 |                   0.7658 |                0.3503 |                0.9891 |            0.7324 |               0.6871 |                          13 |
| lastfm0.03 | test    | pure       |               1024 |                        882 |            3764 |                 0.2343 |                 0.8100 |                   0.8229 |                0.5772 |                0.9943 |            0.8792 |               0.4589 |                         128 |
| lastfm0.03 | valid   | permissive |                104 |                         96 |            3763 |                 0.0255 |                 0.7205 |                   0.7671 |                0.3021 |                0.9888 |            0.7272 |               0.6991 |                          13 |
| lastfm0.03 | valid   | pure       |               1128 |                        952 |            3763 |                 0.2530 |                 0.8065 |                   0.8225 |                0.5586 |                0.9936 |            0.8728 |               0.4737 |                         141 |

Tier dataset sizes:

| dataset    | split   |   subset_rows |   subset_sessions | tier       |   full_rows |   full_sessions |   row_pct_of_split |   session_pct_of_split |
|:-----------|:--------|--------------:|------------------:|:-----------|------------:|----------------:|-------------------:|-----------------------:|
| lastfm0.03 | valid   |         18840 |               952 | pure       |       55816 |            3763 |             0.3375 |                 0.2530 |
| lastfm0.03 | test    |         15392 |               882 | pure       |       49241 |            3764 |             0.3126 |                 0.2343 |
| lastfm0.03 | valid   |          2557 |                96 | permissive |       55816 |            3763 |             0.0458 |                 0.0255 |
| lastfm0.03 | test    |          2356 |                98 | permissive |       49241 |            3764 |             0.0478 |                 0.0260 |

All-tier union dataset size:

| dataset    | split   |   subset_rows |   subset_sessions |   full_rows |   full_sessions |   row_pct_of_split |   session_pct_of_split |
|:-----------|:--------|--------------:|------------------:|------------:|----------------:|-------------------:|-----------------------:|
| lastfm0.03 | valid   |         19801 |               991 |       55816 |            3763 |             0.3548 |                 0.2634 |
| lastfm0.03 | test    |         16476 |               932 |       49241 |            3764 |             0.3346 |                 0.2476 |

## beauty

| dataset   | split   | tier       |   selected_entries |   selected_unique_sessions |   full_sessions |   session_pct_of_split |   selection_score_mean |   selection_score_median |   selection_score_min |   selection_score_max |   core_score_mean |   balance_score_mean |   effective_quota_after_cap |
|:----------|:--------|:-----------|-------------------:|---------------------------:|----------------:|-----------------------:|-----------------------:|-------------------------:|----------------------:|----------------------:|------------------:|---------------------:|----------------------------:|
| beauty    | test    | permissive |                168 |                        150 |             637 |                 0.2355 |                 0.5925 |                   0.5812 |                0.0468 |                0.9713 |            0.6677 |               0.1660 |                          24 |
| beauty    | test    | pure       |                272 |                        228 |             637 |                 0.3579 |                 0.6727 |                   0.6990 |                0.1459 |                0.9768 |            0.7499 |               0.2353 |                          34 |
| beauty    | valid   | permissive |                161 |                        145 |             636 |                 0.2280 |                 0.5937 |                   0.5813 |                0.0481 |                0.9846 |            0.6685 |               0.1697 |                          23 |
| beauty    | valid   | pure       |                208 |                        195 |             636 |                 0.3066 |                 0.6886 |                   0.6990 |                0.1636 |                0.9842 |            0.7655 |               0.2527 |                          26 |

Tier dataset sizes:

| dataset   | split   |   subset_rows |   subset_sessions | tier       |   full_rows |   full_sessions |   row_pct_of_split |   session_pct_of_split |
|:----------|:--------|--------------:|------------------:|:-----------|------------:|----------------:|-------------------:|-----------------------:|
| beauty    | valid   |          1369 |               195 | pure       |        4252 |             636 |             0.3220 |                 0.3066 |
| beauty    | test    |          1350 |               228 | pure       |        3468 |             637 |             0.3893 |                 0.3579 |
| beauty    | valid   |          1061 |               145 | permissive |        4252 |             636 |             0.2495 |                 0.2280 |
| beauty    | test    |           885 |               150 | permissive |        3468 |             637 |             0.2552 |                 0.2355 |

All-tier union dataset size:

| dataset   | split   |   subset_rows |   subset_sessions |   full_rows |   full_sessions |   row_pct_of_split |   session_pct_of_split |
|:----------|:--------|--------------:|------------------:|------------:|----------------:|-------------------:|-----------------------:|
| beauty    | valid   |          2043 |               283 |        4252 |             636 |             0.4805 |                 0.4450 |
| beauty    | test    |          1894 |               324 |        3468 |             637 |             0.5461 |                 0.5086 |

## foursquare

| dataset    | split   | tier       |   selected_entries |   selected_unique_sessions |   full_sessions |   session_pct_of_split |   selection_score_mean |   selection_score_median |   selection_score_min |   selection_score_max |   core_score_mean |   balance_score_mean |   effective_quota_after_cap |
|:-----------|:--------|:-----------|-------------------:|---------------------------:|----------------:|-----------------------:|-----------------------:|-------------------------:|----------------------:|----------------------:|------------------:|---------------------:|----------------------------:|
| foursquare | test    | permissive |               1024 |                        770 |            3806 |                 0.2023 |                 0.7072 |                   0.7616 |                0.2757 |                0.9653 |            0.7658 |               0.3864 |                         128 |
| foursquare | test    | pure       |               1608 |                       1383 |            3806 |                 0.3634 |                 0.7689 |                   0.8225 |                0.4181 |                0.9847 |            0.8416 |               0.3741 |                         201 |
| foursquare | valid   | permissive |               1024 |                        746 |            3805 |                 0.1961 |                 0.7017 |                   0.7373 |                0.2744 |                0.9672 |            0.7628 |               0.3705 |                         128 |
| foursquare | valid   | pure       |               1608 |                       1358 |            3805 |                 0.3569 |                 0.7665 |                   0.8225 |                0.4176 |                0.9791 |            0.8390 |               0.3756 |                         201 |

Tier dataset sizes:

| dataset    | split   |   subset_rows |   subset_sessions | tier       |   full_rows |   full_sessions |   row_pct_of_split |   session_pct_of_split |
|:-----------|:--------|--------------:|------------------:|:-----------|------------:|----------------:|-------------------:|-----------------------:|
| foursquare | valid   |          7281 |              1358 | pure       |       15638 |            3805 |             0.4656 |                 0.3569 |
| foursquare | test    |          7139 |              1383 | pure       |       15223 |            3806 |             0.4690 |                 0.3634 |
| foursquare | valid   |          4988 |               746 | permissive |       15638 |            3805 |             0.3190 |                 0.1961 |
| foursquare | test    |          4828 |               770 | permissive |       15223 |            3806 |             0.3172 |                 0.2023 |

All-tier union dataset size:

| dataset    | split   |   subset_rows |   subset_sessions |   full_rows |   full_sessions |   row_pct_of_split |   session_pct_of_split |
|:-----------|:--------|--------------:|------------------:|------------:|----------------:|-------------------:|-----------------------:|
| foursquare | valid   |          8853 |              1663 |       15638 |            3805 |             0.5661 |                 0.4371 |
| foursquare | test    |          8778 |              1726 |       15223 |            3806 |             0.5766 |                 0.4535 |

## movielens1m

| dataset     | split   | tier       |   selected_entries |   selected_unique_sessions |   full_sessions |   session_pct_of_split |   selection_score_mean |   selection_score_median |   selection_score_min |   selection_score_max |   core_score_mean |   balance_score_mean |   effective_quota_after_cap |
|:------------|:--------|:-----------|-------------------:|---------------------------:|----------------:|-----------------------:|-----------------------:|-------------------------:|----------------------:|----------------------:|------------------:|---------------------:|----------------------------:|
| movielens1m | test    | permissive |               1024 |                        823 |            2181 |                 0.3773 |                 0.6076 |                   0.6122 |                0.1248 |                0.8614 |            0.6915 |               0.1349 |                         128 |
| movielens1m | test    | pure       |               3952 |                       2010 |            2181 |                 0.9216 |                 0.6584 |                   0.6892 |                0.1122 |                0.9942 |            0.7339 |               0.2308 |                         494 |
| movielens1m | valid   | permissive |               1024 |                        852 |            2181 |                 0.3906 |                 0.5881 |                   0.6385 |                0.1301 |                0.8815 |            0.6646 |               0.1554 |                         128 |
| movielens1m | valid   | pure       |               2904 |                       1809 |            2181 |                 0.8294 |                 0.6666 |                   0.7123 |                0.1272 |                0.9950 |            0.7427 |               0.2351 |                         363 |

Tier dataset sizes:

| dataset     | split   |   subset_rows |   subset_sessions | tier       |   full_rows |   full_sessions |   row_pct_of_split |   session_pct_of_split |
|:------------|:--------|--------------:|------------------:|:-----------|------------:|----------------:|-------------------:|-----------------------:|
| movielens1m | valid   |         71464 |              1809 | pure       |       89000 |            2181 |             0.8030 |                 0.8294 |
| movielens1m | test    |         71677 |              2010 | pure       |       79524 |            2181 |             0.9013 |                 0.9216 |
| movielens1m | valid   |         33211 |               852 | permissive |       89000 |            2181 |             0.3732 |                 0.3906 |
| movielens1m | test    |         30190 |               823 | permissive |       79524 |            2181 |             0.3796 |                 0.3773 |

All-tier union dataset size:

| dataset     | split   |   subset_rows |   subset_sessions |   full_rows |   full_sessions |   row_pct_of_split |   session_pct_of_split |
|:------------|:--------|--------------:|------------------:|------------:|----------------:|-------------------:|-----------------------:|
| movielens1m | valid   |         76643 |              1924 |       89000 |            2181 |             0.8612 |                 0.8822 |
| movielens1m | test    |         74744 |              2081 |       79524 |            2181 |             0.9399 |                 0.9541 |

## retail_rocket

| dataset       | split   | tier       |   selected_entries |   selected_unique_sessions |   full_sessions |   session_pct_of_split |   selection_score_mean |   selection_score_median |   selection_score_min |   selection_score_max |   core_score_mean |   balance_score_mean |   effective_quota_after_cap |
|:--------------|:--------|:-----------|-------------------:|---------------------------:|----------------:|-----------------------:|-----------------------:|-------------------------:|----------------------:|----------------------:|------------------:|---------------------:|----------------------------:|
| retail_rocket | test    | permissive |               1024 |                        817 |           22964 |                 0.0356 |                 0.7174 |                   0.6828 |                0.5179 |                0.9479 |            0.7594 |               0.5072 |                         128 |
| retail_rocket | test    | pure       |              10936 |                       8812 |           22964 |                 0.3837 |                 0.7815 |                   0.7950 |                0.5600 |                0.9986 |            0.8845 |               0.2135 |                        1367 |
| retail_rocket | valid   | permissive |               1024 |                        802 |           22964 |                 0.0349 |                 0.7164 |                   0.6797 |                0.5109 |                0.9490 |            0.7586 |               0.5008 |                         128 |
| retail_rocket | valid   | pure       |              10728 |                       8692 |           22964 |                 0.3785 |                 0.7811 |                   0.7945 |                0.5434 |                0.9974 |            0.8842 |               0.2123 |                        1341 |

Tier dataset sizes:

| dataset       | split   |   subset_rows |   subset_sessions | tier       |   full_rows |   full_sessions |   row_pct_of_split |   session_pct_of_split |
|:--------------|:--------|--------------:|------------------:|:-----------|------------:|----------------:|-------------------:|-----------------------:|
| retail_rocket | valid   |         48590 |              8692 | pure       |      105278 |           22964 |             0.4615 |                 0.3785 |
| retail_rocket | test    |         46715 |              8812 | pure       |       99387 |           22964 |             0.4700 |                 0.3837 |
| retail_rocket | valid   |          8737 |               802 | permissive |      105278 |           22964 |             0.0830 |                 0.0349 |
| retail_rocket | test    |          8397 |               817 | permissive |       99387 |           22964 |             0.0845 |                 0.0356 |

All-tier union dataset size:

| dataset       | split   |   subset_rows |   subset_sessions |   full_rows |   full_sessions |   row_pct_of_split |   session_pct_of_split |
|:--------------|:--------|--------------:|------------------:|------------:|----------------:|-------------------:|-----------------------:|
| retail_rocket | valid   |         49080 |              8769 |      105278 |           22964 |             0.4662 |                 0.3819 |
| retail_rocket | test    |         47377 |              8917 |       99387 |           22964 |             0.4767 |                 0.3883 |
