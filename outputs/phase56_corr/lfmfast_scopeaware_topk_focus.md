# LFMFAST Top-k Scope Sanity

Formula used:
- group scope: n_eff_scope_norm = macro_n_eff / expert_scale
- global scope with top-k>0: n_eff_scope_norm = macro_n_eff / moe_top_k
- group normalization: macro_group_n_eff_expscale_norm = macro_group_n_eff / expert_scale

Key rows:
- LFMFAST_C2 lastfm0.03_FeaturedMoE_N3_lfmfast_c2_20260318_084008_203193_pid438110: scope=global_flat, k=6.0, scale=3.0, n_eff=10.6192, n_eff_scope_norm=1.7699, group_n_eff/scale=1.3006
- LFMFAST_C2 lastfm0.03_FeaturedMoE_N3_lfmfast_c2_20260318_134320_272910_pid3926: scope=global_flat, k=6.0, scale=3.0, n_eff=10.4316, n_eff_scope_norm=1.7386, group_n_eff/scale=1.3033
- LFMFAST_C3 lastfm0.03_FeaturedMoE_N3_lfmfast_c3_20260318_134320_302515_pid3927: scope=global_flat, k=3.0, scale=3.0, n_eff=10.6173, n_eff_scope_norm=3.5391, group_n_eff/scale=1.3104
- LFMFAST_C3 lastfm0.03_FeaturedMoE_N3_lfmfast_c3_20260318_084008_234759_pid438111: scope=global_flat, k=3.0, scale=3.0, n_eff=9.9789, n_eff_scope_norm=3.3263, group_n_eff/scale=1.3032
- LFMFAST_C4 lastfm0.03_FeaturedMoE_N3_lfmfast_c4_20260318_100601_827082_pid449463: scope=group_dense, k=0.0, scale=3.0, n_eff=11.4912, n_eff_scope_norm=3.8304, group_n_eff/scale=1.2930
