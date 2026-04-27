# Balanced Tier Summary

## KuaiRecLargeStrictPosV2_0.2

| dataset                     | split   | tier       |   quota |   selected_entries |   selected_unique_sessions |   full_sessions |   session_pct_of_split |   subset_rows |   full_rows |   row_pct_of_split |   selection_score_mean |   selection_score_median |   selection_score_min |   selection_score_max |   core_score_mean |   balance_score_mean |
|:----------------------------|:--------|:-----------|--------:|-------------------:|---------------------------:|----------------:|-----------------------:|--------------:|------------:|-------------------:|-----------------------:|-------------------------:|----------------------:|----------------------:|------------------:|---------------------:|
| KuaiRecLargeStrictPosV2_0.2 | test    | permissive |     113 |                904 |                        709 |            3669 |                 0.1932 |          4657 |       14614 |             0.3187 |                 0.6777 |                   0.6789 |                0.4735 |                0.9932 |            0.7136 |               0.4847 |
| KuaiRecLargeStrictPosV2_0.2 | test    | pure       |      80 |                640 |                        594 |            3669 |                 0.1619 |          3967 |       14614 |             0.2715 |                 0.7907 |                   0.8280 |                0.5496 |                0.9944 |            0.8433 |               0.5027 |
| KuaiRecLargeStrictPosV2_0.2 | valid   | permissive |      62 |                496 |                        449 |            3669 |                 0.1224 |          5858 |       33853 |             0.1730 |                 0.6120 |                   0.6000 |                0.1660 |                0.9907 |            0.6652 |               0.3150 |
| KuaiRecLargeStrictPosV2_0.2 | valid   | pure       |      94 |                752 |                        655 |            3669 |                 0.1785 |          8666 |       33853 |             0.2560 |                 0.7253 |                   0.7410 |                0.1825 |                0.9944 |            0.8064 |               0.2894 |

All-tier union subset:

| dataset                     | split   |   selected_unique_sessions_all_tiers |   full_sessions |   session_pct_of_split |   subset_rows_all_tiers |   full_rows |   row_pct_of_split |
|:----------------------------|:--------|-------------------------------------:|----------------:|-----------------------:|------------------------:|------------:|-------------------:|
| KuaiRecLargeStrictPosV2_0.2 | test    |                                 1133 |            3669 |                 0.3088 |                    6956 |       14614 |             0.4760 |
| KuaiRecLargeStrictPosV2_0.2 | valid   |                                  979 |            3669 |                 0.2668 |                   12318 |       33853 |             0.3639 |

## lastfm0.03

| dataset    | split   | tier       |   quota |   selected_entries |   selected_unique_sessions |   full_sessions |   session_pct_of_split |   subset_rows |   full_rows |   row_pct_of_split |   selection_score_mean |   selection_score_median |   selection_score_min |   selection_score_max |   core_score_mean |   balance_score_mean |
|:-----------|:--------|:-----------|--------:|-------------------:|---------------------------:|----------------:|-----------------------:|--------------:|------------:|-------------------:|-----------------------:|-------------------------:|----------------------:|----------------------:|------------------:|---------------------:|
| lastfm0.03 | test    | permissive |      13 |                104 |                         98 |            3764 |                 0.0260 |          2356 |       49241 |             0.0478 |                 0.7236 |                   0.7658 |                0.3503 |                0.9891 |            0.7324 |               0.6871 |
| lastfm0.03 | test    | pure       |     128 |               1024 |                        882 |            3764 |                 0.2343 |         15392 |       49241 |             0.3126 |                 0.8100 |                   0.8229 |                0.5772 |                0.9943 |            0.8792 |               0.4589 |
| lastfm0.03 | valid   | permissive |      13 |                104 |                         96 |            3763 |                 0.0255 |          2557 |       55816 |             0.0458 |                 0.7205 |                   0.7671 |                0.3021 |                0.9888 |            0.7272 |               0.6991 |
| lastfm0.03 | valid   | pure       |     141 |               1128 |                        952 |            3763 |                 0.2530 |         18840 |       55816 |             0.3375 |                 0.8065 |                   0.8225 |                0.5586 |                0.9936 |            0.8728 |               0.4737 |

All-tier union subset:

| dataset    | split   |   selected_unique_sessions_all_tiers |   full_sessions |   session_pct_of_split |   subset_rows_all_tiers |   full_rows |   row_pct_of_split |
|:-----------|:--------|-------------------------------------:|----------------:|-----------------------:|------------------------:|------------:|-------------------:|
| lastfm0.03 | test    |                                  932 |            3764 |                 0.2476 |                   16476 |       49241 |             0.3346 |
| lastfm0.03 | valid   |                                  991 |            3763 |                 0.2634 |                   19801 |       55816 |             0.3548 |

## beauty

| dataset   | split   | tier       |   quota |   selected_entries |   selected_unique_sessions |   full_sessions |   session_pct_of_split |   subset_rows |   full_rows |   row_pct_of_split |   selection_score_mean |   selection_score_median |   selection_score_min |   selection_score_max |   core_score_mean |   balance_score_mean |
|:----------|:--------|:-----------|--------:|-------------------:|---------------------------:|----------------:|-----------------------:|--------------:|------------:|-------------------:|-----------------------:|-------------------------:|----------------------:|----------------------:|------------------:|---------------------:|
| beauty    | test    | permissive |      24 |                168 |                        150 |             637 |                 0.2355 |           885 |        3468 |             0.2552 |                 0.5925 |                   0.5812 |                0.0468 |                0.9713 |            0.6677 |               0.1660 |
| beauty    | test    | pure       |      34 |                272 |                        228 |             637 |                 0.3579 |          1350 |        3468 |             0.3893 |                 0.6727 |                   0.6990 |                0.1459 |                0.9768 |            0.7499 |               0.2353 |
| beauty    | valid   | permissive |      23 |                161 |                        145 |             636 |                 0.2280 |          1061 |        4252 |             0.2495 |                 0.5937 |                   0.5813 |                0.0481 |                0.9846 |            0.6685 |               0.1697 |
| beauty    | valid   | pure       |      26 |                208 |                        195 |             636 |                 0.3066 |          1369 |        4252 |             0.3220 |                 0.6886 |                   0.6990 |                0.1636 |                0.9842 |            0.7655 |               0.2527 |

All-tier union subset:

| dataset   | split   |   selected_unique_sessions_all_tiers |   full_sessions |   session_pct_of_split |   subset_rows_all_tiers |   full_rows |   row_pct_of_split |
|:----------|:--------|-------------------------------------:|----------------:|-----------------------:|------------------------:|------------:|-------------------:|
| beauty    | test    |                                  324 |             637 |                 0.5086 |                    1894 |        3468 |             0.5461 |
| beauty    | valid   |                                  283 |             636 |                 0.4450 |                    2043 |        4252 |             0.4805 |

## foursquare

| dataset    | split   | tier       |   quota |   selected_entries |   selected_unique_sessions |   full_sessions |   session_pct_of_split |   subset_rows |   full_rows |   row_pct_of_split |   selection_score_mean |   selection_score_median |   selection_score_min |   selection_score_max |   core_score_mean |   balance_score_mean |
|:-----------|:--------|:-----------|--------:|-------------------:|---------------------------:|----------------:|-----------------------:|--------------:|------------:|-------------------:|-----------------------:|-------------------------:|----------------------:|----------------------:|------------------:|---------------------:|
| foursquare | test    | permissive |     302 |               2416 |                       1700 |            3806 |                 0.4467 |          8345 |       15223 |             0.5482 |                 0.6598 |                   0.6322 |                0.2212 |                0.9653 |            0.7191 |               0.3388 |
| foursquare | test    | pure       |     201 |               1608 |                       1383 |            3806 |                 0.3634 |          7139 |       15223 |             0.4690 |                 0.7689 |                   0.8225 |                0.4181 |                0.9847 |            0.8416 |               0.3741 |
| foursquare | valid   | permissive |     338 |               2704 |                       1723 |            3805 |                 0.4528 |          8855 |       15638 |             0.5662 |                 0.6499 |                   0.6268 |                0.1806 |                0.9672 |            0.7096 |               0.3279 |
| foursquare | valid   | pure       |     201 |               1608 |                       1358 |            3805 |                 0.3569 |          7281 |       15638 |             0.4656 |                 0.7665 |                   0.8225 |                0.4176 |                0.9791 |            0.8390 |               0.3756 |

All-tier union subset:

| dataset    | split   |   selected_unique_sessions_all_tiers |   full_sessions |   session_pct_of_split |   subset_rows_all_tiers |   full_rows |   row_pct_of_split |
|:-----------|:--------|-------------------------------------:|----------------:|-----------------------:|------------------------:|------------:|-------------------:|
| foursquare | test    |                                 2417 |            3806 |                 0.6350 |                   11165 |       15223 |             0.7334 |
| foursquare | valid   |                                 2387 |            3805 |                 0.6273 |                   11391 |       15638 |             0.7284 |

## movielens1m

| dataset     | split   | tier       |   quota |   selected_entries |   selected_unique_sessions |   full_sessions |   session_pct_of_split |   subset_rows |   full_rows |   row_pct_of_split |   selection_score_mean |   selection_score_median |   selection_score_min |   selection_score_max |   core_score_mean |   balance_score_mean |
|:------------|:--------|:-----------|--------:|-------------------:|---------------------------:|----------------:|-----------------------:|--------------:|------------:|-------------------:|-----------------------:|-------------------------:|----------------------:|----------------------:|------------------:|---------------------:|
| movielens1m | test    | permissive |     458 |               3664 |                       1853 |            2181 |                 0.8496 |         69539 |       79524 |             0.8744 |                 0.5869 |                   0.6059 |                0.0315 |                0.8614 |            0.6757 |               0.0853 |
| movielens1m | test    | pure       |     494 |               3952 |                       2010 |            2181 |                 0.9216 |         71677 |       79524 |             0.9013 |                 0.6584 |                   0.6892 |                0.1122 |                0.9942 |            0.7339 |               0.2308 |
| movielens1m | valid   | permissive |     441 |               3528 |                       1857 |            2181 |                 0.8514 |         76535 |       89000 |             0.8599 |                 0.5748 |                   0.6363 |                0.0261 |                0.8815 |            0.6578 |               0.1049 |
| movielens1m | valid   | pure       |     363 |               2904 |                       1809 |            2181 |                 0.8294 |         71464 |       89000 |             0.8030 |                 0.6666 |                   0.7123 |                0.1272 |                0.9950 |            0.7427 |               0.2351 |

All-tier union subset:

| dataset     | split   |   selected_unique_sessions_all_tiers |   full_sessions |   session_pct_of_split |   subset_rows_all_tiers |   full_rows |   row_pct_of_split |
|:------------|:--------|-------------------------------------:|----------------:|-----------------------:|------------------------:|------------:|-------------------:|
| movielens1m | test    |                                 2156 |            2181 |                 0.9885 |                   78305 |       79524 |             0.9847 |
| movielens1m | valid   |                                 2121 |            2181 |                 0.9725 |                   86038 |       89000 |             0.9667 |

## retail_rocket

| dataset       | split   | tier       |   quota |   selected_entries |   selected_unique_sessions |   full_sessions |   session_pct_of_split |   subset_rows |   full_rows |   row_pct_of_split |   selection_score_mean |   selection_score_median |   selection_score_min |   selection_score_max |   core_score_mean |   balance_score_mean |
|:--------------|:--------|:-----------|--------:|-------------------:|---------------------------:|----------------:|-----------------------:|--------------:|------------:|-------------------:|-----------------------:|-------------------------:|----------------------:|----------------------:|------------------:|---------------------:|
| retail_rocket | test    | permissive |    1637 |              13096 |                       9014 |           22964 |                 0.3925 |         53796 |       99387 |             0.5413 |                 0.6167 |                   0.6027 |                0.1294 |                0.9479 |            0.6991 |               0.1592 |
| retail_rocket | test    | pure       |    1367 |              10936 |                       8812 |           22964 |                 0.3837 |         46715 |       99387 |             0.4700 |                 0.7815 |                   0.7950 |                0.5600 |                0.9986 |            0.8845 |               0.2135 |
| retail_rocket | valid   | permissive |    1606 |              12848 |                       8979 |           22964 |                 0.3910 |         56276 |      105278 |             0.5345 |                 0.6129 |                   0.5939 |                0.1279 |                0.9490 |            0.6942 |               0.1610 |
| retail_rocket | valid   | pure       |    1341 |              10728 |                       8692 |           22964 |                 0.3785 |         48590 |      105278 |             0.4615 |                 0.7811 |                   0.7945 |                0.5434 |                0.9974 |            0.8842 |               0.2123 |

All-tier union subset:

| dataset       | split   |   selected_unique_sessions_all_tiers |   full_sessions |   session_pct_of_split |   subset_rows_all_tiers |   full_rows |   row_pct_of_split |
|:--------------|:--------|-------------------------------------:|----------------:|-----------------------:|------------------------:|------------:|-------------------:|
| retail_rocket | test    |                                13306 |           22964 |                 0.5794 |                   68529 |       99387 |             0.6895 |
| retail_rocket | valid   |                                13302 |           22964 |                 0.5793 |                   72022 |      105278 |             0.6841 |
