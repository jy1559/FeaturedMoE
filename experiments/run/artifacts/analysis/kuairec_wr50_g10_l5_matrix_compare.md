# KuaiRec WR50 G10 L5 Matrix Compare

Rule:

- `clip(watch_ratio, 1.0) >= 0.5`
- session gap on `next_start - prev_end`
- `gap = 10 min`
- `min_session_len = 5`
- chunk suffix `_c{idx}` with `max_session_len = 50`

| source | source_rows | rows_after_watch_filter | rows_written | users | items | sessions | avg_session_len | median_session_len | p90_session_len |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| big_matrix | 12,530,806 | 8,031,791 | 5,433,801 | 6,754 | 9,601 | 383,972 | 14.15 | 9 | 35 |
| small_matrix | 4,676,570 | 3,401,473 | 1,943,613 | 1,411 | 3,312 | 192,405 | 10.10 | 7 | 18 |

Generated dataset:

- `processed/basic/KuaiRecSmallWR50G10L5`
- `processed/feature_added_v2/KuaiRecSmallWR50G10L5`
