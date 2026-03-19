# P5+P6 Input vs Output Explainer

## 1) Input (control) variables
These are hyperparameters set before training. They are optimization pressures, not observed behavior themselves.

- `balance_loss_lambda`
  - role: penalizes unbalanced expert usage.
  - expected effect: higher value tends to flatten expert usage distribution (lower concentration).
- `z_loss_lambda`
  - role: logit-scale regularizer on router outputs.
  - expected effect: stabilizes routing logits; may reduce extreme gate values.
- `route_smoothness_lambda`
  - role: encourages smoother routing over neighboring positions/sessions.
  - expected effect: less abrupt route switching.
- `route_sharpness_lambda`
  - role: encourages sharper routing decisions.
  - expected effect: more peaked gate distribution.
- `route_prior_lambda`
  - role: aligns routing toward prior preference.
  - expected effect: stronger prior-driven routing.
- `route_monopoly_lambda`
  - role: penalizes monopolistic expert takeover.
  - expected effect: suppresses single-expert dominance.
- `route_consistency_lambda`
  - role: encourages consistency between related samples.
  - expected effect: higher neighborhood consistency.
- `group_prior_align_lambda`
  - role: aligns group-level routing with group prior.
  - expected effect: stronger group-structured specialization.
- `factored_group_balance_lambda`
  - role: group/factored balance regularization.
  - expected effect: balances load across group/factor dimensions.

## 2) Output (observed) variables
These are measured from routing diagnostics after/within training.

- `n_eff_ratio_topkaware`
  - base metric: `n_eff` (effective number of experts used).
  - top-k-aware normalization:
    - group scope: `n_eff / expert_scale`
    - global scope with top-k: `n_eff / moe_top_k`
    - global scope without top-k: `n_eff / expert_count(=12)`
  - interpretation: larger means broader effective expert usage within available routing capacity.

- `cv_usage_mean`
  - base metric: stage `cv_usage` (coefficient of variation over expert usage).
  - aggregation: mean over macro/mid/micro.
  - interpretation: larger means more imbalance/skew in expert usage.

- `top1_over_uniform_mean`
  - base metric: `top1_max_frac` per stage.
  - capacity-aware transform: `top1_max_frac * denominator` (same denominator rule as above).
  - aggregation: mean over macro/mid/micro.
  - interpretation: larger means stronger top-1 concentration relative to a uniform baseline.

- `knn_score_mean`
  - base metric: `route_consistency_knn_score`.
  - aggregation: mean over macro/mid/micro.
  - interpretation: larger means stronger neighborhood routing consistency.

- `group_knn_score_mean`
  - base metric: `route_consistency_group_knn_score`.
  - aggregation: mean over macro/mid/micro.
  - interpretation: larger means stronger group-level neighborhood consistency.

- `intra_group_knn_mean_score_mean`
  - base metric: `route_consistency_intra_group_knn_mean_score`.
  - aggregation: mean over macro/mid/micro.
  - interpretation: larger means stronger within-group routing consistency.

## 3) Why input/output must be separated

- Input metrics are causes/controls (what we set).
- Output metrics are effects/observations (what model does).
- Correlation between input and MRR can be indirect and configuration-dependent.
- Correlation between output and MRR is descriptive of trained behavior, not causal by itself.

## 4) Practical reading guide for your scatter plots

- For `n_eff_ratio_topkaware`:
  - compare only after denominator correction (already applied).
  - this prevents `top6`, `top3`, and `group_top1` runs from being mixed on incompatible scales.
- For concentration-vs-balance:
  - read `top1_over_uniform_mean` together with `cv_usage_mean`.
- For consistency terms:
  - read `knn_score_mean`, `group_knn_score_mean`, `intra_group_knn_mean_score_mean` together.
