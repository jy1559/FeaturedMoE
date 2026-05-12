# hparams

This file is the anchor notebook for future `exp_overall` runs.

The goal is not to name a single historical winner. The goal is to say, for each dataset/model pair, where the next run should start without reopening raw logs or JSON.

## Decision Rules

Use the following order whenever sources disagree.

1. Full-data `CIKM_8` safety evidence wins over sampled historical winners.
2. Otherwise start from `candidate_rank=1` in `artifact_anchor_catalog.csv`.
3. Use `candidate_rank=2..4` only as nearby alternatives, not as the default.
4. Use curated route configs from `selected_configs_final_topk.json` only for structure when raw route coverage is missing or incomplete.

This means full `lastfm` and full `KuaiRec` are handled differently from their sampled historical parents.

## Dataset-Level Default Bands

These are the first anchor bands before going model-by-model.

- `beauty`: default `MAX_ITEM_LIST_LENGTH=10`; allow `20-30` only for `sasrec`, `fdsa`, `fearec`, and `tisasrec`.
- `foursquare`: default `20-30`; allow `50` mainly for `fdsa`.
- `movielens1m`: default `10` for `sasrec`, `gru4rec`, `fdsa`, `difsr`; use `20-30` for `duorec`, `fame`, `tisasrec`.
- `retail_rocket`: feature models mostly like `20-30`; `sasrec` and `gru4rec` tolerate `50`.
- sampled `KuaiRecLargeStrictPosV2_0.2`: most good anchors live in `10-30`; route prefers `20` in the sampled raw runs.
- sampled `lastfm0.03`: quality-oriented sampled winners spread from `10` to `50`, but full `lastfm` should still be run conservatively.

Batch defaults also split by family.

- Standard baselines on `beauty`, `foursquare`, `movielens1m`, `retail_rocket`: usually `4096/4096` or `4096/8192`.
- Heavy feature baselines such as `difsr`, `fdsa`, `tisasrec` on larger datasets: often `1024/2048` or `2048/4096`.
- `featured_moe_n3`: default `2048/2048` unless full-data safety evidence says to shrink further.

## Route Model Transfer Rules

The route model needs special handling because the artifact surface is asymmetric.

- `beauty`, `foursquare`, and sampled `KuaiRec` have usable raw route anchors in `results_final_experiment_fmoe`.
- `movielens1m` has raw route anchors, but the curated route digest is better for optimizer fields.
- `retail_rocket` route raw JSON is missing in the current workspace, so the curated route digest is the only structural source.
- Sampled `lastfm0.03` route winners are useful for structure, but full `lastfm` still uses the conservative `CIKM_8` gate.

When a route anchor is missing in raw artifacts, borrow optimizer scale from nearby route datasets.

- Learning rate band: roughly `5e-4` to `1.3e-3`
- Weight decay band: roughly `5e-7` to `1e-6`
- Hidden size band: roughly `128` to `224`
- `fixed_hidden_dropout_prob`: usually `0.12` to `0.18`
- `route_consistency_lambda`: usually `2.5e-4` to `1.2e-3`

## Full-Data Overrides That Still Win

These are not optional for the current runner.

### Full KuaiRec

- `fdsa`: keep the successful full anchor around `MAX_ITEM_LIST_LENGTH=50`, `train/eval batch=2048/2048`, `attribute_hidden_size=160`, `lambda_attr=0.12`, and low attention dropout.
- `featured_moe_n3`: keep the successful full-safe anchor around `MAX_ITEM_LIST_LENGTH=10`, `train/eval batch=2048/2048`, and low dropout.

Even though sampled `KuaiRecLargeStrictPosV2_0.2` sometimes favors longer lists or larger hidden sizes, the full-data overrides remain the correct starting point in `exp_overall`.

### Full LastFM

For full `lastfm`, the runner should remain conservative.

- Force `show_progress=true`
- Bias toward `MAX_ITEM_LIST_LENGTH=10`
- Bias toward `max_evals=1`
- Prefer fixed-config or near-fixed runs first
- Shrink batches aggressively when the model is heavy

Known safe full anchors from earlier `CIKM_8` work:

- `gru4rec`: `lr` around `0.00139541`, `MAX_ITEM_LIST_LENGTH=10`, `1024/256`
- `sasrec`: `lr` around `0.000972843`, `MAX_ITEM_LIST_LENGTH=10`, `1024/256`
- `tisasrec`: `lr` around `0.000972843`, `MAX_ITEM_LIST_LENGTH=10`, `1024/256`
- `duorec`: `lr` around `0.000697705`, `MAX_ITEM_LIST_LENGTH=10`, `1024/256`

For `bsarec`, `difsr`, `fame`, `fdsa`, and `fearec`, keep their sampled structural anchors but clamp them into the same conservative full-`lastfm` regime.

### Full LastFM Route Gate

`featured_moe_n3` on full `lastfm` remains high risk.

- Use `MAX_ITEM_LIST_LENGTH=10`
- Use `max_evals=1`
- Keep `show_progress=true`
- Start with very small batches such as `512/128`
- Treat the row as preflight-only until it proves stable

## Dataset-By-Dataset Anchor Recommendations

### beauty

The strongest pattern on `beauty` is short lists and medium-width models. Most useful anchors sit at `MAX_ITEM_LIST_LENGTH=10`, and the main exception is attribute-heavy feature models.

- `bsarec`: start from `MAX_ITEM_LIST_LENGTH=10`, `hidden_size=64`, `embedding_size=128`, `num_layers=2`, `num_heads=4`, `lr=0.00127456`, `wd=1e-4`, `bsarec_alpha=0.55`, `bsarec_c=2`.
- `difsr`: start from `MAX_ITEM_LIST_LENGTH=10`, `hidden_size=128`, `num_layers=2`, `num_heads=4`, `attribute_hidden_size=192`, `lambda_attr=0.12`, `selected_features=category`, `lr=0.00099281`, `wd=1e-6`.
- `duorec`: this is one of the cleaner `beauty` anchors. Start from `MAX_ITEM_LIST_LENGTH=10`, `hidden_size=128`, `embedding_size=96`, `n_layers=2`, `n_heads=1`, `tau=0.16`, `lmd=0.05`, `lr=0.00134695`, `wd=1e-5`.
- `fame`: use only when the expert mixture itself matters. Start from `MAX_ITEM_LIST_LENGTH=10`, `hidden_size=64`, `embedding_size=160`, `num_experts=5`, `num_layers=1`, `lr=0.00243852`, `wd=1.504868294019483e-4`.
- `fdsa`: this is one of the better `beauty` families. Start from `MAX_ITEM_LIST_LENGTH=30`, `hidden_size=192`, `embedding_size=96`, `attribute_hidden_size=160`, `lambda_attr=0.15`, `n_layers=2`, `n_heads=2`, `lr=0.000546`, `wd=5e-5`.
- `fearec`: use a conservative anchor because `beauty` is not one of its strongest datasets. Start from `MAX_ITEM_LIST_LENGTH=30`, `hidden_size=112`, `embedding_size=128`, `n_layers=2`, `n_heads=1`, `tau=0.22`, `lmd=0.03`, `lr=0.00082629`, `wd=2.16e-4`.
- `gru4rec`: quality is weaker than `duorec`, `fdsa`, and `tisasrec`, but if you need the family, start from `MAX_ITEM_LIST_LENGTH=10`, `hidden_size=128`, `embedding_size=160`, `dropout_prob=0.25`, `hidden_dropout_prob=0.2`, `lr=0.0040343`, `wd=1.49e-6`.
- `sasrec`: start from `MAX_ITEM_LIST_LENGTH=20`, `hidden_size=64`, `embedding_size=96`, `num_layers=1`, `num_heads=2`, `hidden_dropout_prob=0.15`, `attn_dropout_prob=0.15`, `lr=0.00134164`, `wd=1e-5`.
- `tisasrec`: prefer `20-30` rather than forcing `10`. Start from `MAX_ITEM_LIST_LENGTH=20`, `hidden_size=128`, `n_layers=1`, `n_heads=4`, `time_span=512`, `train/eval batch=2048/4096`, `lr=0.00042722`, `wd=1e-6`.
- `featured_moe_n3`: the best route anchors are compact. Start from `MAX_ITEM_LIST_LENGTH=10`, `hidden_size=96`, `d_ff=192`, `d_expert_hidden=96`, `d_router_hidden=32`, `d_feat_emb=8`, `expert_scale=4`, `hidden_dropout_prob=0.16`, `fixed_hidden_dropout_prob=0.18`, `lr=0.0005672`, `wd=5e-7`, `route_consistency_lambda=0.0005`, `stage_feature_dropout_prob=0.0`.

### foursquare

`foursquare` is the cleanest dataset for medium-length sequence anchors. Many good rows cluster around `MAX_ITEM_LIST_LENGTH=20-30`, with `fdsa` as the main long-list exception.

- `bsarec`: start from `MAX_ITEM_LIST_LENGTH=20`, `hidden_size=64`, `embedding_size=96`, `num_layers=2`, `num_heads=4`, `bsarec_alpha=0.55`, `bsarec_c=2`, `lr=0.00137668`, `wd=1e-4`.
- `difsr`: start from `MAX_ITEM_LIST_LENGTH=20`, `hidden_size=192`, `num_layers=3`, `num_heads=4`, `attribute_hidden_size=96`, `lambda_attr=0.1`, `selected_features=category`, `lr=0.00107236`, `wd=1e-4`.
- `duorec`: one of the strongest stable anchors. Start from `MAX_ITEM_LIST_LENGTH=20`, `hidden_size=128`, `embedding_size=160`, `n_layers=3`, `n_heads=1`, `tau=0.18`, `lmd=0.02`, `lmd_sem=0.02`, `lr=0.00082255`, `wd=0.0`.
- `fame`: keep it small and short. Start from `MAX_ITEM_LIST_LENGTH=10`, `hidden_size=160`, `embedding_size=96`, `num_experts=5`, `num_layers=1`, `num_heads=1`, `lr=0.00113349`, `wd=1e-4`.
- `fdsa`: this family likes longer context here. Start from `MAX_ITEM_LIST_LENGTH=50`, `hidden_size=160`, `embedding_size=160`, `attribute_hidden_size=144`, `lambda_attr=0.1`, `n_layers=4`, `n_heads=8`, `train/eval batch=2048/2048`, `lr=0.00019668`, `wd=0.0`.
- `fearec`: start from `MAX_ITEM_LIST_LENGTH=30`, `hidden_size=128`, `embedding_size=96`, `n_layers=2`, `n_heads=4`, `tau=0.25`, `lmd=0.04`, `lr=0.00056307`, `wd=1.5e-4`.
- `gru4rec`: if you need an RNN anchor, keep the high-lr regime. Start from `MAX_ITEM_LIST_LENGTH=30`, `hidden_size=224`, `embedding_size=160`, `dropout_prob=0.25`, `hidden_dropout_prob=0.23578510204832642`, `lr=0.00621669`, `wd=1e-4`.
- `sasrec`: start from `MAX_ITEM_LIST_LENGTH=30`, `hidden_size=96`, `embedding_size=96`, `num_layers=1`, `num_heads=4`, `hidden_dropout_prob=0.1`, `attn_dropout_prob=0.2`, `lr=0.00059161`, `wd=5e-5`.
- `tisasrec`: start from `MAX_ITEM_LIST_LENGTH=20`, `hidden_size=64`, `n_layers=2`, `n_heads=1`, `time_span=128`, `train/eval batch=1024/2048`, `lr=0.0006606`, `wd=1e-4`.
- `featured_moe_n3`: the best raw route row is a compact but not tiny model. Start from `MAX_ITEM_LIST_LENGTH=10`, `hidden_size=224`, `d_ff=448`, `d_expert_hidden=224`, `d_router_hidden=128`, `d_feat_emb=16`, `expert_scale=4`, `hidden_dropout_prob=0.16`, `fixed_hidden_dropout_prob=0.15`, `lr=0.00049957`, `wd=5e-7`, `route_consistency_lambda=0.0005`, `stage_feature_dropout_prob=0.03`.

### movielens1m

`movielens1m` splits into two groups. Plain sequential models tend to prefer short lists. Contrastive or richer mixture models can stretch to `20-30`.

- `bsarec`: start from `MAX_ITEM_LIST_LENGTH=20`, `hidden_size=64`, `embedding_size=96`, `num_layers=2`, `num_heads=1`, `bsarec_alpha=0.5`, `bsarec_c=2`, `lr=0.00137668`, `wd=1e-5`.
- `difsr`: start from `MAX_ITEM_LIST_LENGTH=10`, `hidden_size=192`, `num_layers=3`, `num_heads=1`, `attribute_hidden_size=96`, `lambda_attr=0.1`, `selected_features=category`, `lr=0.00068518`, `wd=5e-5`.
- `duorec`: start from `MAX_ITEM_LIST_LENGTH=30`, `hidden_size=96`, `embedding_size=112`, `n_layers=2`, `n_heads=4`, `tau=0.22`, `lmd=0.05`, `lr=0.00049613`, `wd=1.0229593556023368e-4`.
- `fame`: start from `MAX_ITEM_LIST_LENGTH=30`, `hidden_size=128`, `embedding_size=160`, `num_experts=4`, `num_layers=2`, `num_heads=2`, `lr=0.0004293`, `wd=1e-4`. If you want the tighter stage2 anchor instead, keep the same shape but switch to `MAX_ITEM_LIST_LENGTH=10` and `lr=0.00075359`.
- `fdsa`: the most reusable anchor is short. Start from `MAX_ITEM_LIST_LENGTH=10`, `hidden_size=128`, `embedding_size=112`, `attribute_hidden_size=192`, `lambda_attr=0.15`, `n_layers=3`, `n_heads=1`, `lr=0.00057119`, `wd=5e-5`.
- `fearec`: keep it conservative. Start from `MAX_ITEM_LIST_LENGTH=20`, `hidden_size=112`, `embedding_size=96`, `n_layers=2`, `n_heads=2`, `tau=0.192`, `lmd=0.04`, `lmd_sem=0.02`, `lr=0.00035026`, `wd=1e-5`.
- `gru4rec`: short context works best. Start from `MAX_ITEM_LIST_LENGTH=10`, `hidden_size=192`, `embedding_size=224`, `dropout_prob=0.2`, `hidden_dropout_prob=0.3`, `lr=0.00405413`, `wd=1e-4`.
- `sasrec`: start from `MAX_ITEM_LIST_LENGTH=10`, `hidden_size=64`, `embedding_size=96`, `num_layers=2`, `num_heads=2`, `hidden_dropout_prob=0.15`, `attn_dropout_prob=0.15`, `lr=0.00092592`, `wd=1e-5`.
- `tisasrec`: this is one of the strongest `movielens1m` anchors. Start from `MAX_ITEM_LIST_LENGTH=30`, `hidden_size=160`, `n_layers=3`, `n_heads=2`, `time_span=256`, `train/eval batch=1024/2048`, `lr=0.00032234`, `wd=1e-5`.
- `featured_moe_n3`: use the curated route digest as the optimizer source and the raw route rows as the width source. Recommended start is `MAX_ITEM_LIST_LENGTH=10`, `hidden_size=128`, `d_ff=256`, `d_expert_hidden=128`, `d_router_hidden=96`, `d_feat_emb=16`, `expert_scale=3`, `hidden_dropout_prob=0.16`, `fixed_hidden_dropout_prob=0.15`, `lr=0.00129037`, `wd=1e-6`, `route_consistency_lambda=0.0012`, `stage_feature_dropout_prob=0.03`, `z_loss_lambda=0.0004`, `train/eval batch=2048/2048`.

### retail_rocket

`retail_rocket` is a strong dataset overall, but its best anchor depends heavily on model family. Plain sequential models tolerate very long lists. Feature models more often settle in `20-30`.

- `bsarec`: start from `MAX_ITEM_LIST_LENGTH=20`, `hidden_size=96`, `embedding_size=160`, `num_layers=1`, `num_heads=4`, `bsarec_alpha=0.5`, `bsarec_c=3`, `lr=0.00137668`, `wd=5e-5`. A lower-lr stage1 alternative at `0.00022945` is also credible once stability is confirmed.
- `difsr`: use the compact-batch feature anchor. Start from `MAX_ITEM_LIST_LENGTH=20`, `hidden_size=128`, `num_layers=2`, `num_heads=2`, `attribute_hidden_size=192`, `lambda_attr=0.1`, `selected_features=category`, `train/eval batch=1024/2048`, `lr=0.00068518`, `wd=1e-6`.
- `duorec`: start from `MAX_ITEM_LIST_LENGTH=20`, `hidden_size=128`, `embedding_size=128`, `n_layers=3`, `n_heads=2`, `tau=0.2`, `lmd=0.06`, `lr=0.00046005`, `wd=1e-5`.
- `fame`: short lists are clearly better than long ones here. Start from `MAX_ITEM_LIST_LENGTH=10`, `hidden_size=160`, `embedding_size=160`, `num_experts=3`, `num_layers=2`, `num_heads=2`, `lr=0.00028161`, `wd=5e-5`.
- `fdsa`: this family wants the smaller-batch feature regime. Start from `MAX_ITEM_LIST_LENGTH=30`, `hidden_size=64`, `embedding_size=128`, `attribute_hidden_size=112`, `lambda_attr=0.0`, `n_layers=2`, `n_heads=1`, `train/eval batch=1024/2048`, `lr=0.00020145`, `wd=5e-5`.
- `fearec`: start from `MAX_ITEM_LIST_LENGTH=30`, `hidden_size=160`, `embedding_size=96`, `n_layers=3`, `n_heads=2`, `tau=0.16`, `lmd=0.05`, `lr=0.00078559`, `wd=5e-5`.
- `gru4rec`: this family likes long lists here. Start from `MAX_ITEM_LIST_LENGTH=50`, `hidden_size=128`, `embedding_size=224`, `dropout_prob=0.15`, `hidden_dropout_prob=0.2`, `lr=0.0042987`, `wd=1e-6`.
- `sasrec`: one of the cleanest anchors on the dataset. Start from `MAX_ITEM_LIST_LENGTH=50`, `hidden_size=96`, `embedding_size=128`, `num_layers=1`, `num_heads=2`, `hidden_dropout_prob=0.2`, `attn_dropout_prob=0.1`, `lr=0.00092592`, `wd=1e-6`.
- `tisasrec`: start from `MAX_ITEM_LIST_LENGTH=30`, `hidden_size=160`, `n_layers=2`, `n_heads=2`, `time_span=384`, `train/eval batch=1024/2048`, `lr=0.0006606`, `wd=1e-6`.
- `featured_moe_n3`: raw route JSON is missing in this workspace snapshot, so use the curated route digest as the structural anchor. Start from `MAX_ITEM_LIST_LENGTH=20`, `hidden_size=192`, `d_ff=384`, `d_expert_hidden=192`, `d_router_hidden=64`, `d_feat_emb=8`, `expert_scale=3`, `hidden_dropout_prob=0.16`, `fixed_hidden_dropout_prob=0.14`, `lr=0.00056966`, `wd=1e-6`, `route_consistency_lambda=0.00025`, `stage_feature_dropout_prob=0.03`, `z_loss_lambda=0.0002` to `0.0004`, `train/eval batch=2048/2048`.

### sampled KuaiRec -> full `KuaiRec`

The artifact catalog here comes from `KuaiRecLargeStrictPosV2_0.2`, so use it as a structural prior, then respect the full-data overrides where they exist.

- `bsarec`: sampled structure says `MAX_ITEM_LIST_LENGTH=30`, `hidden_size=128`, `embedding_size=128`, `num_layers=2`, `num_heads=1`, `bsarec_alpha=0.7`, `lr=0.00294347`, `wd=5e-5`.
- `difsr`: sampled structure says `MAX_ITEM_LIST_LENGTH=30`, `hidden_size=96`, `num_layers=1`, `num_heads=4`, `attribute_hidden_size=192`, `lambda_attr=0.1`, `lr=0.00229281`, `wd=1e-6`.
- `duorec`: sampled structure says `MAX_ITEM_LIST_LENGTH=20`, `hidden_size=128`, `embedding_size=96`, `n_layers=2`, `n_heads=1`, `tau=0.16`, `lmd=0.06`, `lr=0.00061408`, `wd=1e-5`.
- `fame`: sampled structure says `MAX_ITEM_LIST_LENGTH=20`, `hidden_size=64`, `embedding_size=96`, `num_experts=4`, `num_heads=8`, `num_layers=2`, `lr=0.0009897`, `wd=1e-4`.
- `fdsa`: sampled structure says `MAX_ITEM_LIST_LENGTH=30`, `hidden_size=112`, `embedding_size=128`, `attribute_hidden_size=80`, `lambda_attr=0.0`, `lr=0.00129969`, `wd=1e-6`, but the full-data `CIKM_8` override still wins for the current runner.
- `fearec`: this is one of the strongest sampled baselines. Start from `MAX_ITEM_LIST_LENGTH=30`, `hidden_size=160`, `embedding_size=128`, `n_layers=1`, `n_heads=1`, `tau=0.2`, `lmd=0.04`, `lr=0.00039416`, `wd=1.5e-4`.
- `gru4rec`: sampled anchors are weaker here. If needed, use `MAX_ITEM_LIST_LENGTH=10`, `hidden_size=96`, `embedding_size=192`, `dropout_prob=0.15`, `hidden_dropout_prob=0.2`, `lr=0.00139555`, `wd=1e-4`.
- `sasrec`: sampled structure says `MAX_ITEM_LIST_LENGTH=10`, `hidden_size=96`, `embedding_size=96`, `num_layers=3`, `num_heads=1`, `hidden_dropout_prob=0.2`, `attn_dropout_prob=0.12`, `lr=0.00126491`, `wd=1e-5`.
- `tisasrec`: sampled structure says `MAX_ITEM_LIST_LENGTH=10`, `hidden_size=64`, `n_layers=2`, `n_heads=2`, `time_span=512`, `train/eval batch=1536/3072`, `lr=0.00202199`, `wd=5e-5`.
- `featured_moe_n3`: sampled raw route winner says `MAX_ITEM_LIST_LENGTH=20`, `hidden_size=256`, `d_ff=512`, `d_expert_hidden=256`, `d_router_hidden=128`, `d_feat_emb=16`, `expert_scale=5`, `hidden_dropout_prob=0.14`, `fixed_hidden_dropout_prob=0.15`, `lr=0.00072438`, `wd=8e-7`, `route_consistency_lambda=0.00025`, `stage_feature_dropout_prob=0.03`, `z_loss_lambda=0.0002`, but the full-data `CIKM_8` route override still wins for the current runner.

### sampled `lastfm0.03` -> full `lastfm`

The sampled artifact catalog is useful for structural direction, but full `lastfm` should still be launched conservatively.

- `bsarec`: sampled quality anchor is `MAX_ITEM_LIST_LENGTH=10`, `hidden_size=64`, `embedding_size=128`, `num_layers=2`, `num_heads=2`, `bsarec_alpha=0.35`, `bsarec_c=5`, `lr=0.00073587`, `wd=1e-5`.
- `difsr`: sampled quality anchor is `MAX_ITEM_LIST_LENGTH=30`, `hidden_size=128`, `num_layers=1`, `num_heads=1`, `attribute_hidden_size=96`, `lambda_attr=0.14`, `selected_features=category`, `lr=0.0005732`, `wd=5e-5`.
- `duorec`: sampled quality anchor is `MAX_ITEM_LIST_LENGTH=20`, `hidden_size=96`, `embedding_size=112`, `n_layers=1`, `n_heads=1`, `tau=0.2`, `lmd=0.05`, `lr=0.00012372`, `wd=1.5e-4`, but full `lastfm` should still be clamped into the conservative full-data batch regime.
- `fame`: sampled quality anchor is `MAX_ITEM_LIST_LENGTH=30`, `hidden_size=64`, `embedding_size=64`, `num_experts=3`, `num_layers=3`, `num_heads=4`, `lr=0.00026706`, `wd=1e-4`.
- `fdsa`: sampled quality anchor is `MAX_ITEM_LIST_LENGTH=20`, `hidden_size=112`, `embedding_size=160`, `attribute_hidden_size=96`, `lambda_attr=0.16`, `n_layers=3`, `n_heads=2`, `lr=0.00050234`, `wd=1e-6`.
- `fearec`: sampled quality anchor is `MAX_ITEM_LIST_LENGTH=50`, `hidden_size=128`, `embedding_size=96`, `n_layers=3`, `n_heads=2`, `tau=0.2`, `lmd=0.03`, `lr=0.00052422`, `wd=1e-5`.
- `gru4rec`: sampled quality anchor is `MAX_ITEM_LIST_LENGTH=20`, `hidden_size=256`, `embedding_size=160`, `dropout_prob=0.1`, `hidden_dropout_prob=0.3`, `lr=0.0029`, `wd=1e-5`.
- `sasrec`: sampled quality anchor is `MAX_ITEM_LIST_LENGTH=20`, `hidden_size=128`, `embedding_size=64`, `num_layers=1`, `num_heads=1`, `hidden_dropout_prob=0.2`, `attn_dropout_prob=0.12`, `lr=0.00049492`, `wd=1e-5`.
- `tisasrec`: sampled quality anchor is `MAX_ITEM_LIST_LENGTH=10`, `hidden_size=64`, `n_layers=2`, `n_heads=2`, `time_span=512`, `train/eval batch=1536/3072`, `lr=0.00202199`, `wd=5e-5`, but full `lastfm` should still be shrunk toward the known safe regime.
- `featured_moe_n3`: sampled raw route rows favor `MAX_ITEM_LIST_LENGTH=30`, `hidden_size=168`, `d_ff=336`, `d_expert_hidden=168`, `d_router_hidden=96`, `d_feat_emb=16`, `expert_scale=4`, `hidden_dropout_prob=0.15`, `fixed_hidden_dropout_prob=0.15`, `lr=0.00052318`, `wd=1e-6`. The curated route digest also points to a larger `224/448` variant around `lr=0.0004983`. For full `lastfm`, do not launch either directly without the full-data safety gate.

## If Only One Anchor Per Pair Is Needed

When budget is tight, the first anchors worth trusting most are:

- `beauty`: `duorec`, `fdsa`, `tisasrec`, `featured_moe_n3`
- `foursquare`: `featured_moe_n3`, `fdsa`, `sasrec`, `tisasrec`
- `movielens1m`: `tisasrec`, `fdsa`, `bsarec`, then route with curated optimizer
- `retail_rocket`: `sasrec`, `difsr`, `fame`, then route with curated structure
- sampled `KuaiRec`: `fearec`, `duorec`, `difsr`, then route; for full `KuaiRec`, keep the explicit `fdsa` and route overrides
- sampled `lastfm`: `fdsa`, `difsr`, `sasrec`, route structure only; for full `lastfm`, keep the conservative gate first

## Bottom Line

The most important operational lesson is still the same one that showed up during the archaeology pass: `MAX_ITEM_LIST_LENGTH` is a real runtime control, not a cosmetic hyperparameter. On `lastfm` especially, use it conservatively first and only widen it after a stable anchor is confirmed.