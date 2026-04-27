# Q4 Experiment Guide

## Purpose

This document explains the current main-text Q4 for RouteRec.

Q4 is no longer an efficiency-first experiment.
It is now a RouteRec-specific robustness and reliance experiment built around the paper's portability claim for behavioral cues.

The target question is:

"Does RouteRec remain effective when only lighter, more portable cues are available, and does the trained model actually rely on those cues at evaluation time?"

This makes Q4 complementary to the other main questions:

- Q2 asks what should control routing.
- Q3 asks why the current routed structure is the right design.
- Q4 asks whether cue-guided routing still works under weaker or corrupted cue conditions.
- Q5 asks what the learned routing behavior responds to.

## Why Q4 was changed

The older Q4 focused on sparse efficiency and active-parameter matching.
That was useful as supporting evidence, but it was too generic and too easy to read as a standard sparse-MoE practicality section.

The new Q4 is more paper-aligned because it directly tests Challenge C1 from the paper:

- behavioral cues should be portable across datasets
- behavioral cues should still be informative enough to guide routing

## High-level design

Q4 uses a `1 + 3 + 3` structure.

Interpretation:

- `1` fixed reference model condition
- `3` training-time cue-availability variations
- `3` evaluation-time cue-efficacy perturbations

Operationally, the code implements this as two linked panels.

## Panel A: Reduced-Cue Availability

Panel A retrains RouteRec under four cue-availability settings.

- `full`
- `remove_category`
- `remove_time`
- `portable_core`

These are defined in `q4_portability_settings()` in `common.py`.

### What each setting means

- `full`
  Uses the default cue configuration with all currently enabled cue families.

- `remove_category`
  Drops group or category-like cues.
  Current keyword drop rule:
  `cat`, `theme`, `overlap`, `mismatch`

- `remove_time`
  Drops timing or pace-like cues.
  Current keyword drop rule:
  `gap`, `pace`, `int`, `age`, `delta`

- `portable_core`
  Drops both category-like and time-like cues, leaving the most portable core subset.

### What this panel is testing

Panel A is asking whether RouteRec still retains quality when richer cue families are unavailable.

This is not an evaluation-time corruption test.
It is a retraining test.
Each variant is trained and selected under its own reduced cue definition.

### What changes and what stays fixed

Changed:

- cue availability via `stage_feature_drop_keywords`

Fixed:

- model family: `featured_moe_n3`
- base candidate source
- dataset split protocol
- selection priority: `overall_seen_target`
- default final training budget intent: `100` epochs, `10` patience
- narrow LR-centered search policy

## Panel B: Fixed-Model Cue Efficacy

Panel B does not retrain additional models.

It takes one completed `full` checkpoint from Panel A and evaluates it under four cue conditions.

- `intact`
- `zero_all`
- `position_permute`
- `cross_sample_permute`

These are defined in `q4_feature_efficacy_specs()` in `common.py`.

### What each intervention means

- `intact`
  Normal evaluation with no perturbation.
  Uses `feature_perturb_mode=none`.

- `zero_all`
  Zeroes the cue input at evaluation time.
  Uses `feature_perturb_mode=zero` and `feature_perturb_apply=eval`.

- `position_permute`
  Permutes cue positions within a sample.
  This breaks within-sequence cue alignment while keeping the sample itself intact.
  Uses `feature_perturb_mode=batch_permute`.

- `cross_sample_permute`
  Permutes cue values across samples.
  This keeps the global cue distribution but attaches cues to the wrong examples.
  Uses `feature_perturb_mode=global_permute`.

### What this panel is testing

Panel B is asking whether the trained RouteRec model actually uses cue input at evaluation time.

This is important because Panel A alone cannot distinguish between two stories:

- RouteRec genuinely relies on cue input at inference time.
- cue-informed training helps, but the final model is not very sensitive to cue corruption at inference time.

Panel B separates those stories.

## How the pipeline is wired

### Main entrypoints

- `q4_portability.py`
- `q4_portability.sh`
- `q4_eval_feature_efficacy.py`
- `export_q2_q5_bundle.py`

### Main flow

1. `q4_portability.py` builds Panel A training rows from `q4_portability_settings()`.
2. It writes a Q4 manifest under the `q4_portability` axis.
3. It runs the training jobs.
4. If the run is not a dry-run and there are successful `full` rows, it starts Panel B postprocessing.
5. By default, it selects one successful `full` checkpoint per dataset.
6. If `--postprocess-all` is used, it runs Panel B for every completed `full` row.
7. `q4_eval_feature_efficacy.py` evaluates the selected checkpoint under the four perturbation conditions.
8. `export_q2_q5_bundle.py` builds the paper-friendly CSVs.

### Default shell launcher

`q4_portability.sh` currently defaults to:

- datasets: `KuaiRecLargeStrictPosV2_0.2,foursquare`
- models: `featured_moe_n3`
- top-k configs: `1`
- seeds: `1`
- GPUs: `0`
- max-evals: `5`
- max-run-hours: `1.0`
- tune-epochs: `100`
- tune-patience: `10`
- LR mode: `narrow_loguniform`
- search algo: `tpe`

These defaults are intentionally narrow enough for a focused paper experiment rather than a broad retuning sweep.

## Storage layout

### Panel A logs

Root:

- `experiments/run/artifacts/logs/real_final_ablation/q4_portability`

Important files:

- `manifest.json`
  Planned run list for the current invocation.

- `summary.csv`
  Main run-level table with status, metrics, checkpoint path, result path, and metadata.

### Panel B logs

Also under:

- `experiments/run/artifacts/logs/real_final_ablation/q4_portability`

Important files:

- `q4_efficacy_index.csv`
  Index from selected full-cue training rows to the per-bundle intervention manifest.

- `efficacy/.../q4_efficacy_manifest.csv`
  Per-bundle manifest written by `q4_eval_feature_efficacy.py`.
  This should normally contain one row per intervention.

- `efficacy/.../logging/.../q4_efficacy_result.json`
  Per-intervention evaluation result payload.

### Paper export files

Root:

- `writing/260419_real_final_exp/data`

Important Q4 exports:

- `q4_portability_table.csv`
- `q4_feature_efficacy.csv`
- `q_suite_manifest.json`
- `q_suite_run_index.csv`

Compatibility note:

- `q4_efficiency_table.csv` is currently written as an alias of `q4_portability_table.csv` for compatibility with older downstream references.

## How to read the outputs

### Panel A export: `q4_portability_table.csv`

Main columns:

- `dataset`
- `setting_key`
- `setting_label`
- `variant_order`
- `base_rank`
- `seed_id`
- `best_valid_seen_mrr20`
- `test_seen_mrr20`
- `retention_vs_full`
- `status`

Interpretation:

- `test_seen_mrr20` is the main concise scalar for paper discussion.
- `retention_vs_full` is the most important derived number for portability claims.
- `status` must be checked before reading the row as evidence.

### Panel B export: `q4_feature_efficacy.csv`

Main columns:

- `dataset`
- `base_rank`
- `seed_id`
- `intervention`
- `intervention_label`
- `best_valid_seen_mrr20`
- `test_seen_mrr20`
- `delta_vs_intact`
- `status`

Interpretation:

- `delta_vs_intact` is the key value for the figure.
- Larger drop means the trained model was more dependent on correctly aligned cue input.

## Current evaluation rule

Unless the user explicitly changes the reporting view, Q4 should be interpreted with `overall_seen_target` first.

Default concise scalar:

- seen-target `MRR@20`

Supporting views such as `overall` or `overall_unseen_target` are secondary and should not drive the main Q4 claim by default.

## Recommended visualizations

### Figure 1: Panel A retention plot

Recommended form:

- grouped bar chart or slope chart
- x-axis: `Full`, `No Group`, `No Time`, `Portable Core`
- y-axis: `test_seen_mrr20` or retention relative to `Full`
- one group or line per dataset

Why this works:

- it shows whether the model degrades gracefully
- it is easy to compare across datasets
- it makes the portability story visible without a large table

### Figure 2: Panel B efficacy plot

Recommended form:

- grouped bars of `delta_vs_intact`
- x-axis: `Intact`, `Zero All`, `Intra-Sequence Permute`, `Cross-Sample Permute`
- y-axis: drop in seen-target `MRR@20`
- one group per dataset

Why this works:

- it directly tests cue reliance at evaluation time
- it avoids overcommitting to route-weight visualizations
- it makes the corruption story easy to explain in one paragraph

### Optional compact table

If a table is needed in addition to figures, keep it minimal.

Suggested Panel A table:

- dataset
- full
- no group
- no time
- portable core
- retention of the strongest reduced-cue setting

Suggested Panel B table:

- dataset
- intact
- zero all
- intra-sequence permute
- cross-sample permute
- max drop vs intact

## What this Q4 should claim

Good claim shape:

- RouteRec retains useful performance even when richer cue families are removed.
- The final trained model loses quality when cue input is corrupted at evaluation time.
- Therefore, RouteRec is not only train-time feature-assisted; it is inference-time cue-sensitive.

## What this Q4 should not claim

Avoid claiming any of the following unless separately validated:

- stronger efficiency than competing sparse models
- stronger total-parameter fairness conclusions
- route-map interpretability based only on weak visual separation
- strong unseen-target generalization from this Q4 alone

Those belong to other analyses or appendix support, not the core Q4 claim.

## Operational notes for future agents

### Summary contamination

Historical Q4 development left old rows in `q4_portability/summary.csv`.
Current export code filters Q4 to the active four settings only:

- `full`
- `remove_category`
- `remove_time`
- `portable_core`

Do not reintroduce old settings such as `sequence_only` into the main export unless the paper design is explicitly changed.

### Postprocess expectation

When Panel B runs correctly, future agents should verify:

- `q4_efficacy_index.csv` exists
- each selected full-cue anchor points to an intervention manifest
- each intervention manifest contains the expected intervention rows
- `q4_feature_efficacy.csv` is non-empty after export

### Cheap validation checklist

Before trusting a Q4 run, check the following:

1. `manifest.json` contains the intended dataset x setting grid.
2. `summary.csv` contains only the intended four Panel A settings for current planned rows.
3. at least one `full` row finished with `status=ok`.
4. `q4_efficacy_index.csv` exists after non-dry run postprocessing.
5. `q4_feature_efficacy.csv` contains `intact`, `zero_all`, `position_permute`, and `cross_sample_permute` rows.
6. exported Q4 tables report seen-target metrics, not only generic overall metrics.

## Suggested next-step workflow

If a future agent needs to continue Q4 work, the most reliable order is:

1. run or resume `q4_portability.py`
2. inspect `q4_portability/summary.csv`
3. confirm at least one successful `full` checkpoint per dataset
4. confirm Panel B artifacts under `q4_portability/efficacy`
5. rerun `export_q2_q5_bundle.py`
6. inspect `q4_portability_table.csv` and `q4_feature_efficacy.csv`
7. only then update paper text or figures

## Bottom line

The current Q4 is a two-part cue study.

Panel A tests whether RouteRec survives weaker cue availability.
Panel B tests whether the trained model actually depends on correctly aligned cue input.

That combination is the main reason this Q4 belongs in the paper.