# MovieLens V4 SessionFixed Portfolio Summary

Date: 2026-04-17

## Current portfolio readout

- Completed result files: 16 / 16
- Practical conclusion: the remaining ML16 branch is not a keep candidate even though its result file exists.
- Baseline context from prior repo notes: baseline_2 movielens1m valid MRR@20 is about 0.0966, so this axis still trails baseline.

## Top validation runs

1. ML07_h6_feat_router_combo: valid 0.0937, test 0.0574
2. ML05_h6_router_up: valid 0.0934, test 0.0585
3. ML04_h6_feat_up: valid 0.0915, test 0.0572
4. ML15_h1_highlr_reframe: valid 0.0905, test 0.0584
5. ML03_h6_expert_up: valid 0.0904, test 0.0587

## Top test runs

1. ML13_h14_capacity_recheck: valid 0.0880, test 0.0606
2. ML03_h6_expert_up: valid 0.0904, test 0.0587
3. ML05_h6_router_up: valid 0.0934, test 0.0585
4. ML15_h1_highlr_reframe: valid 0.0905, test 0.0584

## Readout

- The main signal is still H6. Router width and feature width helped more than plain lr-only refinement.
- ML05 and ML07 are the safest follow-up anchors because they jointly maximize validation while keeping test near the portfolio top tier.
- ML03 is worth keeping because expert_scale=3 gives the best H6-side test result.
- ML13 is the only clearly interesting non-H6 survivor on test, but it looks like a higher-variance branch rather than the new default anchor.
- ML16_h3_midaux_probe ended at valid 0.0866 and test 0.0524, so it is not a strong continuation candidate.

## MovieLens_2 redesign

Baseline read-across from movielens1m now points to two stronger regimes than the original `_2` bank captured:

- Compact mid-width runs around hidden 160, max_len 30, 3-layer, 4-head, dropout about 0.15, weight decay about 2e-4.
- Lightweight high-regularization runs around hidden 112, max_len 50, 1-layer, 4-head, dropout about 0.25, weight decay about 5e-4.

That means the earlier `_2` direction was too centered on H6/H14 capacity and too lightly regularized. The updated bank keeps lr-only tuning but moves the fixed point closer to BSARec/FDSA-style movielens scales.

Conservative lr-only refinement:

1. ML201_h5_compact30_core
2. ML202_h5_compact30_feat16
3. ML203_h4_light50_core
4. ML204_h7_compact30_core

Aggressive lr-only combos:

1. ML205_h5_compact30_feat16_router96
2. ML206_h7_compact40_feat16_expert4
3. ML207_h4_light50_feat16_regularized
4. ML208_h6_bridge30_router84

All eight MovieLens_2 runs still tune only learning_rate, but the fixed hyperparameters now use longer lengths, materially larger regularization, lower aux loss, and more compact capacity targets.