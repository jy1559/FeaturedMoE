# KuaiRec sampling study

| method | ratio | interactions | sessions | users | items | avg_session_len | avg_sessions_per_user | avg_interactions_per_user | preserve_score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full | 1.00 | 12,510,424 | 828,542 | 7,176 | 10,724 | 15.10 | 115.46 | 1743.37 | 0.0000 |
| user_stratified | 0.03 | 375,091 | 25,241 | 215 | 8,550 | 14.86 | 117.40 | 1744.61 | 0.0338 |
| session_stratified | 0.03 | 374,840 | 24,855 | 6,632 | 8,534 | 15.08 | 3.75 | 56.52 | 2.5466 |
| chrono_head | 0.03 | 375,315 | 43,306 | 6,799 | - | 8.67 | 6.37 | 55.20 | 3.6563 |
| interaction_resession | 0.03 | 29,837 | 4,455 | 2,646 | 3,832 | 6.70 | 1.68 | 11.28 | 3.6896 |
| chrono_tail | 0.03 | 375,320 | 39,416 | 6,353 | - | 9.52 | 6.20 | 59.08 | 3.7730 |

Lower `preserve_score` is better.