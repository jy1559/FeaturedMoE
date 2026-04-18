# Server C Route Plan

This document describes the `server_c` FeaturedMoE-only experiment flow for `final_experiment`.

## Goal

Use a three-stage route-only pipeline that is:

- broad enough in Stage 1 to preserve reviewer-facing grid-search coverage
- biased toward historically strong families so convergence is faster
- automatically narrowed by previous-stage results without manual editing

## Stage Design

| Stage | What is fixed | What is searched | How candidates are chosen |
| --- | --- | --- | --- |
| Stage 1 | One family fixes `embedding_size`, `d_ff`, `d_expert_hidden`, `d_feat_emb`, `hidden_dropout_prob`, stage-family dropout | `learning_rate`, `weight_decay`, `MAX_ITEM_LIST_LENGTH`, `expert_scale`, `d_router_hidden`, `stage_feature_dropout_prob`, `attn_dropout_prob`, `route_consistency_lambda`, `z_loss_lambda` | Family bank mixes history anchors, preset anchors, and neighbor probes |
| Stage 2 | Keep the Stage 1 family capacity fixed | Stage 1 search axes, plus local refinement on `d_feat_emb` and `hidden_dropout_prob` | Top Stage 1 families per dataset, plus one history challenger on datasets with reliable prior signal |
| Stage 3 | Full config fixed | No search, `max_evals=1` | Top Stage 2 configs by validation mean, plus optional history challenger when top-two margin is small |

## Stage 1 Bank Policy

Stage 1 families are tagged as:

- `history_anchor`: directly supported by previous FMoE runs
- `preset_anchor`: known good capacity anchors from the existing FMoE bank
- `neighbor_probe`: small variations used to preserve diversity

This keeps the search broad, but prevents Server C from spending most of its budget on obviously weak families.

## Stage 2 Automatic Narrowing

When `stage2_focus_search.py` runs for `featured_moe_n3`, it now:

1. reads Stage 1 results from `stage1/summary.csv`
2. keeps the top Stage 1 winner families per dataset
3. injects one history challenger family on datasets with strong prior signal
4. builds a smaller inner search space around the best Stage 1 trials

Stage 2 refinement adds local search around:

- `d_feat_emb`
- `hidden_dropout_prob`

while still refining:

- `learning_rate`
- `weight_decay`
- `MAX_ITEM_LIST_LENGTH`
- `expert_scale`
- `d_router_hidden`
- `stage_feature_dropout_prob`
- `attn_dropout_prob`
- `route_consistency_lambda`
- `z_loss_lambda`

## Stage 3 Automatic Selection

When `stage3_seed_confirm.py` runs for `featured_moe_n3`, it:

1. reads Stage 2 results
2. selects the top Stage 2 configs by validation mean
3. adds one history challenger only when the top-two margin is small
4. reruns the selected configs with the dataset seed budget

This gives us a cleaner final selection rule while still protecting against unstable winners.

## Server C Wrapper

The recommended wrapper entrypoint is:

```bash
bash experiments/run/final_experiment/run_server_c.sh pipeline
```

`pipeline` runs:

1. `stage1-fast`
2. `stage2-fast`
3. `stage1-slow`
4. `stage2-slow`
5. `stage2` safety pass over all datasets
6. `stage3`

The stage summaries and manifests are now merged instead of overwritten, so split runs still feed the next stage correctly.
