# Router Wrapper Comparison (KuaiRec)

- Scope: existing `fmoe_n3` wrapper-related experiments on `KuaiRecLargeStrictPosV2_0.2`
- Main clean comparison: `phase8_router_wrapper_diag_v1` Stage A, where wrapper family changes while bias/profile stay fixed
- Follow-up check: `phase8_2_verification_v1` bases `A-D`

## P8 Wrapper-Only Ranking

| Rank(V) | Wrapper | Wrapper Map | Best Valid MRR@20 | Test MRR@20 | Gap(Test-Valid) | Rank(T) | Mean Valid | Mean Test | Runs |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `all_w2` | macro=w2_a_plus_d, mid=w2_a_plus_d, micro=w2_a_plus_d | 0.0819 | 0.1603 | 0.0784 | 6 | 0.0812 | 0.1608 | 2 |
| 2 | `all_w5` | macro=w5_exd, mid=w5_exd, micro=w5_exd | 0.0819 | 0.1602 | 0.0783 | 7 | 0.0813 | 0.1610 | 2 |
| 3 | `mixed_1` | macro=w4_bxd, mid=w4_bxd, micro=w1_flat | 0.0814 | 0.1587 | 0.0773 | 9 | 0.0810 | 0.1601 | 2 |
| 4 | `mixed_2` | macro=w4_bxd, mid=w6_bxd_plus_a, micro=w1_flat | 0.0813 | 0.1604 | 0.0791 | 5 | 0.0809 | 0.1604 | 2 |
| 5 | `all_w4` | macro=w4_bxd, mid=w4_bxd, micro=w4_bxd | 0.0810 | 0.1613 | 0.0803 | 4 | 0.0809 | 0.1614 | 2 |
| 6 | `all_w6` | macro=w6_bxd_plus_a, mid=w6_bxd_plus_a, micro=w6_bxd_plus_a | 0.0809 | 0.1599 | 0.0790 | 8 | 0.0808 | 0.1608 | 2 |
| 7 | `all_w3` | macro=w3_bxc, mid=w3_bxc, micro=w3_bxc | 0.0805 | 0.1613 | 0.0808 | 3 | 0.0804 | 0.1613 | 2 |
| 8 | `mixed_3` | macro=w6_bxd_plus_a, mid=w1_flat, micro=w1_flat | 0.0803 | 0.1617 | 0.0814 | 1 | 0.0803 | 0.1617 | 1 |
| 9 | `all_w1` | macro=w1_flat, mid=w1_flat, micro=w1_flat | 0.0801 | 0.1615 | 0.0814 | 2 | 0.0799 | 0.1617 | 3 |

Interpretation: `mixed_2` is the best valid-side wrapper-only candidate, while all wrappers are tightly clustered on test MRR around ~0.161x in this phase.

## P8_2 Verification Bases

| Rank(V) | Base | Meaning | Best Valid MRR@20 | Test MRR@20 | Gap(Test-Valid) | Rank(T) | Mean Valid | Mean Test | Runs |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `D` | all_w2 + bias_off + src_base | 0.0820 | 0.1594 | 0.0774 | 4 | 0.0811 | 0.1608 | 16 |
| 2 | `B` | mixed_2 + bias_group_feat + src_base | 0.0819 | 0.1598 | 0.0779 | 2 | 0.0808 | 0.1609 | 16 |
| 3 | `C` | all_w5 + bias_off + src_base | 0.0818 | 0.1599 | 0.0781 | 1 | 0.0811 | 0.1609 | 16 |
| 4 | `A` | all_w5 + bias_rule + src_abc_feature | 0.0814 | 0.1595 | 0.0781 | 3 | 0.0811 | 0.1608 | 16 |

Interpretation: `B = mixed_2 + bias_group_feat + src_base` stayed competitive in the verification stage and is the lineage that later became the default final wrapper map.

## Current Final Evidence

- Current `Final_all_datasets` best-valid Kuai run: `A9/H14/s1` with valid `0.1719`, test `0.1691`.
- These final-family runs use `macro=w4_bxd / mid=w6_bxd_plus_a / micro=w1_flat` as the production default wrapper map.

## Takeaway

- If you want one unified wrapper map, the strongest existing evidence still points to `mixed_2 = macro w4_bxd / mid w6_bxd_plus_a / micro w1_flat`.
- Pure `all_w6` did not clearly beat `mixed_2` on valid-side ranking in the clean wrapper-only phase.
- `all_w4` is a reasonable simpler fallback, but existing experiments suggest the `mid=w6_bxd_plus_a` correction is at least worth keeping.
- If you want to truly unify everything to a single wrapper family, the missing comparison is a fresh `all_w6` or `all_w4` rerun under the same P14/A6~A9 budget, because the existing clean wrapper tests are older Kuai-only experiments.
