# ProtoX P1_fast_wide

## Overview
- Entry script: experiments/run/fmoe_protox/run_protox_group.sh
- Order: ML1 R1 -> RR R1 -> ML1 R2 -> RR R2
- Round1 budget: combos_per_gpu=20, max_eval=10, epochs=25, patience=10
- Round2 rule: top_ratio=0.5, scale=1.5, eval/epochs rounded to multiples of 5

## Fixed Params
- embedding_size=160, num_heads=8, d_feat_emb=16, expert_scale=3
- proto_num=8, proto_pooling=query, moe_top_k=0
- temperatures: macro=1.0, mid=1.3, micro=1.3

## Search Ranges
- learning_rate=[5e-5,5e-2] (loguniform)
- weight_decay=[0.0,1e-7,1e-6,1e-5,1e-4,5e-4,1e-3,5e-3] (choice)
- hidden_dropout_prob=[0.0,0.05,0.1,0.15,0.2,0.25,0.3] (choice)
- balance_loss_lambda=[0.001,0.003,0.01,0.03,0.05,0.1,0.2] (choice)
- proto_usage_lambda=[0.0,1e-5,1e-4,3e-4,1e-3,3e-3,1e-2] (choice)
- proto_entropy_lambda=[0.0,1e-5,1e-4,3e-4,1e-3,3e-3,1e-2] (choice)

## Round1 Combo Catalog (10 layout + 10 dim/temp)
| idx | tag | group | merge | gpre | gpost | macro | mid | micro | d_exp | d_router | proto_dim | top_k | temp_start | temp_end | floor | delta |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | L00 | layout | serial_weighted | 0 | 0 | 0 | 0 | 0 | 160 | 80 | 48 | 0 | 1.2 | 1.0 | 0.0 | 1.0 |
| 2 | L01 | layout | serial_weighted | 0 | 1 | 0 | 0 | 0 | 160 | 80 | 48 | 0 | 1.2 | 1.0 | 0.0 | 1.0 |
| 3 | L02 | layout | parallel_weighted | 0 | 0 | 0 | 0 | 0 | 160 | 80 | 48 | 0 | 1.2 | 1.0 | 0.0 | 1.0 |
| 4 | L03 | layout | serial_weighted | 1 | 0 | 0 | 0 | 0 | 160 | 80 | 48 | 0 | 1.1 | 0.9 | 0.0 | 1.0 |
| 5 | L04 | layout | parallel_weighted | 1 | 0 | 0 | 0 | 0 | 160 | 80 | 48 | 0 | 1.1 | 0.9 | 0.0 | 1.0 |
| 6 | L05 | layout | serial_weighted | 0 | 0 | 1 | 0 | 0 | 160 | 80 | 48 | 0 | 1.2 | 1.0 | 0.0 | 1.0 |
| 7 | L06 | layout | serial_weighted | 0 | 0 | 0 | 1 | 0 | 160 | 80 | 48 | 0 | 1.2 | 1.0 | 0.0 | 1.0 |
| 8 | L07 | layout | serial_weighted | 0 | 0 | 0 | 0 | 1 | 160 | 80 | 48 | 0 | 1.2 | 1.0 | 0.0 | 1.0 |
| 9 | L08 | layout | parallel_weighted | 0 | 0 | 1 | 1 | 0 | 160 | 80 | 48 | 0 | 1.1 | 0.9 | 0.0 | 1.0 |
| 10 | L09 | layout | serial_weighted | 0 | 2 | 0 | 1 | 1 | 160 | 80 | 48 | 0 | 1.1 | 0.9 | 0.0 | 1.0 |
| 11 | D00 | dim-temp | serial_weighted | 0 | 1 | 0 | 1 | 1 | 128 | 64 | 32 | 2 | 1.2 | 1.0 | 0.0 | 1.0 |
| 12 | D01 | dim-temp | serial_weighted | 0 | 1 | 0 | 1 | 1 | 160 | 80 | 48 | 2 | 1.1 | 0.9 | 0.05 | 1.5 |
| 13 | D02 | dim-temp | serial_weighted | 0 | 1 | 0 | 1 | 1 | 192 | 96 | 64 | 2 | 1.0 | 0.8 | 0.05 | 1.5 |
| 14 | D03 | dim-temp | parallel_weighted | 1 | 0 | 1 | 1 | 0 | 192 | 96 | 64 | 2 | 0.9 | 0.7 | 0.1 | 2.0 |
| 15 | D04 | dim-temp | parallel_weighted | 1 | 0 | 1 | 1 | 0 | 256 | 96 | 48 | 0 | 1.2 | 1.0 | 0.0 | 1.0 |
| 16 | D05 | dim-temp | serial_weighted | 0 | 2 | 0 | 1 | 1 | 320 | 112 | 64 | 2 | 1.0 | 0.8 | 0.1 | 1.5 |
| 17 | D06 | dim-temp | serial_weighted | 0 | 2 | 0 | 1 | 1 | 384 | 128 | 64 | 2 | 0.9 | 0.7 | 0.1 | 2.0 |
| 18 | D07 | dim-temp | parallel_weighted | 1 | 1 | 1 | 1 | 1 | 512 | 128 | 64 | 0 | 1.1 | 0.9 | 0.05 | 1.0 |
| 19 | D08 | dim-temp | serial_weighted | 0 | 2 | 0 | 1 | 1 | 256 | 80 | 32 | 2 | 1.2 | 0.9 | 0.05 | 2.0 |
| 20 | D09 | dim-temp | parallel_weighted | 1 | 1 | 1 | 1 | 1 | 192 | 112 | 48 | 2 | 1.0 | 0.75 | 0.1 | 2.0 |

## RR Mix Rule
- RR Round1: 80% independent combos + 20% ML1 top-derived variants
- If ML1 top extraction fails, RR Round1 uses 100% independent catalog

## Summary Files
- summary_movielens1m.csv / .md
- summary_retail_rocket.csv / .md
