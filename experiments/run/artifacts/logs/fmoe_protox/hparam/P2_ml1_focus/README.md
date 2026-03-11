# ProtoX P2_ml1_focus

## Overview
- Entry script: experiments/run/fmoe_protox/run_protox_group.sh
- Order: ML1 R1 -> RR R1 -> ML1 R2 -> RR R2 (datasets not requested are skipped)
- Round1 budget: combos_per_gpu=8, max_eval=15, epochs=40, patience=15
- Round2 rule: top_ratio=0.5, scale=1.25, eval/epochs rounded to multiples of 5

## Fixed Params
- embedding_size=160, num_heads=8, d_feat_emb=16, expert_scale=3
- proto_num=8, proto_pooling=query, moe_top_k=0
- temperatures: macro=1.0, mid=1.3, micro=1.3
- stage token correction: enabled (scale=0.5)

## Search Ranges
- learning_rate=[2e-4,2e-2] (loguniform)
- weight_decay=[0.0,1e-7,1e-6,1e-5,1e-4] (choice)
- hidden_dropout_prob=[0.0,0.05,0.1,0.15,0.2,0.25] (choice)
- balance_loss_lambda=[0.01,0.03,0.05,0.1,0.2] (choice)
- proto_usage_lambda=[0.0,1e-5,1e-4,3e-4,1e-3] (choice)
- proto_entropy_lambda=[0.0,1e-4,3e-4,1e-3,3e-3,1e-2] (choice)

## Round1 Combo Catalog
| idx | tag | group | merge | gpre | gpost | macro | mid | micro | d_exp | d_router | proto_dim | top_k | temp_start | temp_end | floor | delta |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | S01 | serial | serial_weighted | 0 | 1 | 0 | 1 | 1 | 160 | 80 | 48 | 2 | 1.1 | 0.9 | 0.05 | 1.5 |
| 2 | S02 | serial | serial_weighted | 0 | 1 | 0 | 1 | 1 | 192 | 96 | 64 | 2 | 1.0 | 0.8 | 0.05 | 1.5 |
| 3 | S03 | serial | serial_weighted | 0 | 1 | 0 | 1 | 1 | 512 | 128 | 64 | 2 | 0.9 | 0.7 | 0.10 | 2.0 |
| 4 | S04 | serial | serial_weighted | 0 | 2 | 0 | 1 | 1 | 160 | 80 | 48 | 2 | 1.1 | 0.9 | 0.05 | 1.5 |
| 5 | S05 | serial | serial_weighted | 0 | 2 | 0 | 1 | 1 | 320 | 112 | 64 | 2 | 1.0 | 0.8 | 0.10 | 1.5 |
| 6 | P01 | parallel | parallel_weighted | 1 | 0 | 1 | 1 | 0 | 192 | 96 | 64 | 2 | 0.9 | 0.7 | 0.10 | 2.0 |
| 7 | P02 | parallel | parallel_weighted | 1 | 0 | 1 | 1 | 0 | 160 | 80 | 48 | 2 | 1.0 | 0.8 | 0.05 | 1.5 |
| 8 | P03 | parallel | parallel_weighted | 1 | 0 | 1 | 1 | 0 | 512 | 128 | 64 | 2 | 0.9 | 0.7 | 0.10 | 2.0 |

## RR Mix Rule
- RR Round1: 80% independent combos + 20% ML1 top-derived variants
- If ML1 top extraction fails, RR Round1 uses 100% independent catalog

## Summary Files
- summary_movielens1m.csv / .md
- summary_retail_rocket.csv / .md
