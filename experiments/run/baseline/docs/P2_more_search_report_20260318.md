# Baseline P2 More Search Report (2026-03-18)

## 1) Summary of completed experiments so far

- Source scripts: `p1_target5_fixed_lr.sh`, `p2_more_search.sh`
- Core datasets: KuaiRecLargeStrictPosV2_0.2, lastfm0.03
- Main metric: test MRR@20

### P1 best by dataset/model

| Dataset | Model | Best test MRR@20 | Best LR | Run phase | Result file |
|---|---:|---:|---:|---|---|
| KuaiRecLargeStrictPosV2_0.2 | BSARec | 0.1573 | 0.0035027831115447498 | P1_FIXREG_W1_D00_BSAREC_C2 | KuaiRecLargeStrictPosV2_0.2_BSARec_p1_fixreg_w1_d00_bsarec_c2_20260314_160911_151935_pid310800.json |
| KuaiRecLargeStrictPosV2_0.2 | FENRec | 0.1585 | 0.007350930134350298 | P1_FIXREG_W1_D00_FENREC_C3 | KuaiRecLargeStrictPosV2_0.2_FENRec_p1_fixreg_w1_d00_fenrec_c3_20260314_220843_747534_pid388982.json |
| KuaiRecLargeStrictPosV2_0.2 | PAtt | 0.1585 | 0.005996537195176188 | P1_FIXREG_W2_D00_PATT_C5 | KuaiRecLargeStrictPosV2_0.2_PAtt_p1_fixreg_w2_d00_patt_c5_20260316_035316_523758_pid483800.json |
| KuaiRecLargeStrictPosV2_0.2 | FAME | 0.1586 | 0.004824408526980862 | P1_FIXREG_W2_D00_FAME_C5 | KuaiRecLargeStrictPosV2_0.2_FAME_p1_fixreg_w2_d00_fame_c5_20260316_050941_857709_pid510817.json |
| KuaiRecLargeStrictPosV2_0.2 | SIGMA | 0.1593 | 0.0007231292937489568 | P1_FIXREG_W2_D00_SIGMA_C7 | KuaiRecLargeStrictPosV2_0.2_SIGMA_p1_fixreg_w2_d00_sigma_c7_20260317_051437_351325_pid134057.json |
| lastfm0.03 | BSARec | 0.3723 | 0.0002240763293798039 | P1_FIXREG_W1_D01_BSAREC_C1 | lastfm0.03_BSARec_p1_fixreg_w1_d01_bsarec_c1_20260314_212211_372299_pid388713.json |
| lastfm0.03 | FENRec | 0.3823 | 0.000953244964078717 | P1_FIXREG_W1_D01_FENREC_C1 | lastfm0.03_FENRec_p1_fixreg_w1_d01_fenrec_c1_20260315_141910_150004_pid400380.json |
| lastfm0.03 | PAtt | 0.3785 | 0.0011520220399198044 | P1_FIXREG_W1_D01_PATT_C1 | lastfm0.03_PAtt_p1_fixreg_w1_d01_patt_c1_20260315_025547_126269_pid398608.json |
| lastfm0.03 | FAME | 0.3732 | 0.00023228825081113833 | P1_FIXREG_W1_D01_FAME_C1 | lastfm0.03_FAME_p1_fixreg_w1_d01_fame_c1_20260315_074505_124233_pid399638.json |
| lastfm0.03 | SIGMA | 0.3809 | 0.0006176568684254302 | P1_FIXREG_W1_D01_SIGMA_C1 | lastfm0.03_SIGMA_p1_fixreg_w1_d01_sigma_c1_20260315_190356_773039_pid409002.json |

### P2 current progress (KuaiRecLargeStrictPosV2_0.2 only)

| Model | Completed combos | Best test MRR@20 | Best LR | Best phase | Completed combo IDs |
|---|---:|---:|---:|---|---|
| BSAREC | 7/16 | 0.1568 | 0.0023113184374976624 | P2_MORE_SEARCH_M00_BSAREC_C06 | C01,C02,C03,C06,C07,C11,C15 |
| FENREC | 4/16 | NA | NA | NA | C03,C07,C11,C15 |
| PATT | 1/16 | NA | NA | NA | C03 |
| FAME | 0/16 | NA | NA | NA | - |
| SIGMA | 0/16 | NA | NA | NA | - |

## 2) P2 combo design and LR strategy

- Queue policy: 4 GPUs, round-robin by combo index (C01/C05/C09/C13 on GPU0, ...).
- Search budget: max-evals=5 per combo for fast filtering.
- Outliers intentionally included: C13 (low edge), C14 (high edge), C15/C16 (mixed edge).

## 3) FENRec matmul dimension issue and fix

- Root cause: FENRec computes contrastive matmul between `forward()` output and hidden view.
- When `hidden_size != embedding_size`, CL logits can hit shape mismatch (matmul runtime error).
- Applied fix in `p2_more_search.sh`:
  - enforce `embedding_size=hidden_size` for all FENRec combos at run command level
  - add pre-run validation: if `hidden_size % num_heads != 0`, stop combo before launch

## 4) Detailed combo specs (model x C01..C16)

### BSARec

| Combo | hidden | layers | heads | extra knobs | LR range | outlier |
|---|---:|---:|---:|---|---|---|
| C01 | 128 | 2 | 4 | ++MAX_ITEM_LIST_LENGTH=20 | 4e-4|3e-3 | no |
| C02 | 160 | 2 | 4 | ++MAX_ITEM_LIST_LENGTH=30 | 4e-4|3e-3 | no |
| C03 | 192 | 3 | 4 | ++MAX_ITEM_LIST_LENGTH=30 | 4e-4|3e-3 | no |
| C04 | 256 | 2 | 8 | ++MAX_ITEM_LIST_LENGTH=30 | 4e-4|3e-3 | no |
| C05 | 128 | 3 | 4 | ++MAX_ITEM_LIST_LENGTH=40 | 7e-4|4e-3 | no |
| C06 | 160 | 1 | 8 | ++MAX_ITEM_LIST_LENGTH=20 | 7e-4|4e-3 | no |
| C07 | 192 | 2 | 8 | ++MAX_ITEM_LIST_LENGTH=40 | 7e-4|4e-3 | no |
| C08 | 224 | 3 | 8 | ++MAX_ITEM_LIST_LENGTH=40 | 7e-4|4e-3 | no |
| C09 | 96 | 2 | 2 | ++MAX_ITEM_LIST_LENGTH=20 | 2.5e-4|2e-3 | no |
| C10 | 144 | 4 | 4 | ++MAX_ITEM_LIST_LENGTH=30 | 2.5e-4|2e-3 | no |
| C11 | 176 | 1 | 4 | ++MAX_ITEM_LIST_LENGTH=50 | 2.5e-4|2e-3 | no |
| C12 | 256 | 4 | 8 | ++MAX_ITEM_LIST_LENGTH=20 | 2.5e-4|2e-3 | no |
| C13 | 80 | 1 | 2 | ++MAX_ITEM_LIST_LENGTH=10 | 1.5e-5|9e-4 | yes |
| C14 | 288 | 3 | 8 | ++MAX_ITEM_LIST_LENGTH=50 | 2e-3|9e-3 | yes |
| C15 | 128 | 4 | 8 | ++MAX_ITEM_LIST_LENGTH=30 | 1e-4|6e-3 | yes |
| C16 | 192 | 2 | 2 | ++MAX_ITEM_LIST_LENGTH=50 | 1e-4|6e-3 | yes |

### FENRec

| Combo | hidden | layers | heads | extra knobs | LR range | outlier |
|---|---:|---:|---:|---|---|---|
| C01 | 128 | 2 | 4 | cl_weight=0.10; cl_temperature=0.10; ++MAX_ITEM_LIST_LENGTH=20 | 6e-4|4e-3 | no |
| C02 | 160 | 2 | 4 | cl_weight=0.15; cl_temperature=0.15; ++MAX_ITEM_LIST_LENGTH=30 | 6e-4|4e-3 | no |
| C03 | 192 | 3 | 8 | cl_weight=0.20; cl_temperature=0.10; ++MAX_ITEM_LIST_LENGTH=30 | 6e-4|4e-3 | no |
| C04 | 224 | 3 | 8 | cl_weight=0.25; cl_temperature=0.20; ++MAX_ITEM_LIST_LENGTH=40 | 6e-4|4e-3 | no |
| C05 | 128 | 3 | 4 | cl_weight=0.30; cl_temperature=0.25; ++MAX_ITEM_LIST_LENGTH=40 | 1e-3|5e-3 | no |
| C06 | 160 | 1 | 4 | cl_weight=0.08; cl_temperature=0.15; ++MAX_ITEM_LIST_LENGTH=20 | 1e-3|5e-3 | no |
| C07 | 192 | 2 | 8 | cl_weight=0.18; cl_temperature=0.08; ++MAX_ITEM_LIST_LENGTH=40 | 1e-3|5e-3 | no |
| C08 | 256 | 2 | 8 | cl_weight=0.25; cl_temperature=0.25; ++MAX_ITEM_LIST_LENGTH=30 | 1e-3|5e-3 | no |
| C09 | 96 | 2 | 2 | cl_weight=0.05; cl_temperature=0.20; ++MAX_ITEM_LIST_LENGTH=20 | 3e-4|2.5e-3 | no |
| C10 | 144 | 4 | 4 | cl_weight=0.22; cl_temperature=0.12; ++MAX_ITEM_LIST_LENGTH=30 | 3e-4|2.5e-3 | no |
| C11 | 176 | 1 | 4 | cl_weight=0.12; cl_temperature=0.30; ++MAX_ITEM_LIST_LENGTH=50 | 3e-4|2.5e-3 | no |
| C12 | 256 | 4 | 8 | cl_weight=0.35; cl_temperature=0.08; ++MAX_ITEM_LIST_LENGTH=20 | 3e-4|2.5e-3 | no |
| C13 | 80 | 1 | 2 | cl_weight=0.03; cl_temperature=0.35; ++MAX_ITEM_LIST_LENGTH=10 | 1.2e-5|7e-4 | yes |
| C14 | 288 | 3 | 8 | cl_weight=0.40; cl_temperature=0.05; ++MAX_ITEM_LIST_LENGTH=50 | 3e-3|1e-2 | yes |
| C15 | 128 | 4 | 8 | cl_weight=0.28; cl_temperature=0.18; ++MAX_ITEM_LIST_LENGTH=30 | 2e-4|6e-3 | yes |
| C16 | 192 | 2 | 2 | cl_weight=0.10; cl_temperature=0.25; ++MAX_ITEM_LIST_LENGTH=50 | 2e-4|6e-3 | yes |

### PAtt

| Combo | hidden | layers | heads | extra knobs | LR range | outlier |
|---|---:|---:|---:|---|---|---|
| C01 | 128 | 2 | 4 | diversity_gamma=0.10; ++MAX_ITEM_LIST_LENGTH=20 | 7e-4|4e-3 | no |
| C02 | 160 | 2 | 4 | diversity_gamma=0.15; ++MAX_ITEM_LIST_LENGTH=30 | 7e-4|4e-3 | no |
| C03 | 192 | 3 | 8 | diversity_gamma=0.20; ++MAX_ITEM_LIST_LENGTH=30 | 7e-4|4e-3 | no |
| C04 | 224 | 3 | 8 | diversity_gamma=0.25; ++MAX_ITEM_LIST_LENGTH=40 | 7e-4|4e-3 | no |
| C05 | 128 | 3 | 4 | diversity_gamma=0.30; ++MAX_ITEM_LIST_LENGTH=40 | 1e-3|5e-3 | no |
| C06 | 160 | 1 | 4 | diversity_gamma=0.05; ++MAX_ITEM_LIST_LENGTH=20 | 1e-3|5e-3 | no |
| C07 | 192 | 2 | 8 | diversity_gamma=0.22; ++MAX_ITEM_LIST_LENGTH=40 | 1e-3|5e-3 | no |
| C08 | 256 | 2 | 8 | diversity_gamma=0.28; ++MAX_ITEM_LIST_LENGTH=30 | 1e-3|5e-3 | no |
| C09 | 96 | 2 | 2 | diversity_gamma=0.08; ++MAX_ITEM_LIST_LENGTH=20 | 4e-4|2.2e-3 | no |
| C10 | 144 | 4 | 4 | diversity_gamma=0.18; ++MAX_ITEM_LIST_LENGTH=30 | 4e-4|2.2e-3 | no |
| C11 | 176 | 1 | 4 | diversity_gamma=0.03; ++MAX_ITEM_LIST_LENGTH=50 | 4e-4|2.2e-3 | no |
| C12 | 256 | 4 | 8 | diversity_gamma=0.35; ++MAX_ITEM_LIST_LENGTH=20 | 4e-4|2.2e-3 | no |
| C13 | 80 | 1 | 2 | diversity_gamma=0.01; ++MAX_ITEM_LIST_LENGTH=10 | 1.5e-5|8e-4 | yes |
| C14 | 288 | 3 | 8 | diversity_gamma=0.40; ++MAX_ITEM_LIST_LENGTH=50 | 3.5e-3|1e-2 | yes |
| C15 | 128 | 4 | 8 | diversity_gamma=0.24; ++MAX_ITEM_LIST_LENGTH=30 | 2e-4|6e-3 | yes |
| C16 | 192 | 2 | 2 | diversity_gamma=0.12; ++MAX_ITEM_LIST_LENGTH=50 | 2e-4|6e-3 | yes |

### FAME

| Combo | hidden | layers | heads | extra knobs | LR range | outlier |
|---|---:|---:|---:|---|---|---|
| C01 | 128 | 2 | 4 | num_experts=4; ++MAX_ITEM_LIST_LENGTH=20 | 5e-4|3.5e-3 | no |
| C02 | 160 | 2 | 4 | num_experts=6; ++MAX_ITEM_LIST_LENGTH=30 | 5e-4|3.5e-3 | no |
| C03 | 192 | 3 | 8 | num_experts=8; ++MAX_ITEM_LIST_LENGTH=30 | 5e-4|3.5e-3 | no |
| C04 | 224 | 3 | 8 | num_experts=10; ++MAX_ITEM_LIST_LENGTH=40 | 5e-4|3.5e-3 | no |
| C05 | 128 | 3 | 4 | num_experts=12; ++MAX_ITEM_LIST_LENGTH=40 | 8e-4|5.5e-3 | no |
| C06 | 160 | 1 | 4 | num_experts=4; ++MAX_ITEM_LIST_LENGTH=20 | 8e-4|5.5e-3 | no |
| C07 | 192 | 2 | 8 | num_experts=10; ++MAX_ITEM_LIST_LENGTH=40 | 8e-4|5.5e-3 | no |
| C08 | 256 | 2 | 8 | num_experts=12; ++MAX_ITEM_LIST_LENGTH=30 | 8e-4|5.5e-3 | no |
| C09 | 96 | 2 | 2 | num_experts=2; ++MAX_ITEM_LIST_LENGTH=20 | 3e-4|2.5e-3 | no |
| C10 | 144 | 4 | 4 | num_experts=8; ++MAX_ITEM_LIST_LENGTH=30 | 3e-4|2.5e-3 | no |
| C11 | 176 | 1 | 4 | num_experts=6; ++MAX_ITEM_LIST_LENGTH=50 | 3e-4|2.5e-3 | no |
| C12 | 256 | 4 | 8 | num_experts=12; ++MAX_ITEM_LIST_LENGTH=20 | 3e-4|2.5e-3 | no |
| C13 | 80 | 1 | 2 | num_experts=2; ++MAX_ITEM_LIST_LENGTH=10 | 1.2e-5|7e-4 | yes |
| C14 | 288 | 3 | 8 | num_experts=16; ++MAX_ITEM_LIST_LENGTH=50 | 3e-3|1e-2 | yes |
| C15 | 128 | 4 | 8 | num_experts=10; ++MAX_ITEM_LIST_LENGTH=30 | 2e-4|6e-3 | yes |
| C16 | 192 | 2 | 2 | num_experts=6; ++MAX_ITEM_LIST_LENGTH=50 | 2e-4|6e-3 | yes |

### SIGMA

| Combo | hidden | layers | heads | extra knobs | LR range | outlier |
|---|---:|---:|---:|---|---|---|
| C01 | 128 | 2 | - | state_size=16; conv_kernel=4; remaining_ratio=0.5; ++MAX_ITEM_LIST_LENGTH=20 | 4e-4|3e-3 | no |
| C02 | 160 | 2 | - | state_size=16; conv_kernel=8; remaining_ratio=0.6; ++MAX_ITEM_LIST_LENGTH=30 | 4e-4|3e-3 | no |
| C03 | 192 | 3 | - | state_size=32; conv_kernel=8; remaining_ratio=0.7; ++MAX_ITEM_LIST_LENGTH=30 | 4e-4|3e-3 | no |
| C04 | 224 | 3 | - | state_size=32; conv_kernel=8; remaining_ratio=0.8; ++MAX_ITEM_LIST_LENGTH=40 | 4e-4|3e-3 | no |
| C05 | 128 | 3 | - | state_size=32; conv_kernel=4; remaining_ratio=0.9; ++MAX_ITEM_LIST_LENGTH=40 | 7e-4|5e-3 | no |
| C06 | 160 | 1 | - | state_size=16; conv_kernel=4; remaining_ratio=0.5; ++MAX_ITEM_LIST_LENGTH=20 | 7e-4|5e-3 | no |
| C07 | 192 | 2 | - | state_size=32; conv_kernel=8; remaining_ratio=0.7; ++MAX_ITEM_LIST_LENGTH=40 | 7e-4|5e-3 | no |
| C08 | 256 | 2 | - | state_size=32; conv_kernel=8; remaining_ratio=0.8; ++MAX_ITEM_LIST_LENGTH=30 | 7e-4|5e-3 | no |
| C09 | 96 | 2 | - | state_size=8; conv_kernel=4; remaining_ratio=0.4; ++MAX_ITEM_LIST_LENGTH=20 | 2e-4|2e-3 | no |
| C10 | 144 | 4 | - | state_size=16; conv_kernel=4; remaining_ratio=0.6; ++MAX_ITEM_LIST_LENGTH=30 | 2e-4|2e-3 | no |
| C11 | 176 | 1 | - | state_size=32; conv_kernel=8; remaining_ratio=0.9; ++MAX_ITEM_LIST_LENGTH=50 | 2e-4|2e-3 | no |
| C12 | 256 | 4 | - | state_size=32; conv_kernel=8; remaining_ratio=0.9; ++MAX_ITEM_LIST_LENGTH=20 | 2e-4|2e-3 | no |
| C13 | 80 | 1 | - | state_size=8; conv_kernel=2; remaining_ratio=0.3; ++MAX_ITEM_LIST_LENGTH=10 | 1e-5|7e-4 | yes |
| C14 | 288 | 3 | - | state_size=48; conv_kernel=8; remaining_ratio=0.95; ++MAX_ITEM_LIST_LENGTH=50 | 2.5e-3|9e-3 | yes |
| C15 | 128 | 4 | - | state_size=32; conv_kernel=8; remaining_ratio=0.85; ++MAX_ITEM_LIST_LENGTH=30 | 2e-4|6e-3 | yes |
| C16 | 192 | 2 | - | state_size=8; conv_kernel=2; remaining_ratio=0.4; ++MAX_ITEM_LIST_LENGTH=50 | 2e-4|6e-3 | yes |

