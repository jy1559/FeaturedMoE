# Scope-Aware Normalization Fix (top-k + expert_scale)

This note applies the denominator rule requested by user feedback:

- global scope with top-k: `n_eff_scope_norm = n_eff / moe_top_k`
- global scope without top-k (`moe_top_k=0`): `n_eff_scope_norm = n_eff / expert_count`
- group scope (`group_*`): `n_eff_scope_norm = n_eff / expert_scale`
- group-local metric: `group_n_eff_expscale_norm = group_n_eff / expert_scale`

Files generated:
- `outputs/phase56_corr/phase56_diag_join_enriched_v3_scopeaware.csv`
- `outputs/phase56_corr/phase56_scopeaware_corr_v2.csv`
- `outputs/phase56_corr/lfmfast_scopeaware_topk_focus.csv`
- `outputs/phase56_corr/lfmfast_scopeaware_topk_focus.md`

Sanity checks from LFMFAST top-k runs:
- `LFMFAST_C2` uses `moe_top_k=6`, `topk_scope_mode=global_flat` (normalizes by 6)
- `LFMFAST_C3` uses `moe_top_k=3`, `topk_scope_mode=global_flat` (normalizes by 3)
- `LFMFAST_C8` uses `topk_scope_mode=group_top1_pergroup`, `expert_scale=3` (group denominator 3)
- `LFMFAST_C9` uses `topk_scope_mode=group_top2_pergroup`, `expert_scale=3` (group denominator 3)

Updated P6 (phase56 join) Spearman with corrected scope handling:
- all: `n_eff=+0.168`, `group_n_eff=+0.674`, `top1=-0.119`, `group_top1=-0.608`
- global_flat only: `n_eff=-0.449`, `group_n_eff=-0.004`
- group_scope only: `n_eff=+0.666`, `group_n_eff=+0.666`

Interpretation:
- The strongest stable signal is in group-aware metrics (`group_n_eff`, `group_top1`).
- Mixing global and group scopes without denominator correction can create misleading clusters (e.g., 0.25/0.5 bands).
- For claims about routing quality, compare within scope (global_flat vs group_scope) or use group-normalized metrics directly.
