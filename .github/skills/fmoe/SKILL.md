---
name: fmoe
description: 'RouteRec and sequential recommendation workflow for this repository. Use when working on fmoe_n4 or baseline_2 experiments, feature_added_v4 datasets, seen-target-first evaluation, paper-aligned result review from writing/ACM_template/sample-sigconf.tex, staged tuning, OOM or missing-output diagnosis, and experiment recommendations for the current paper.'
argument-hint: 'Task for fmoe_n4, baseline_2, paper-aligned result review, run planning, or diagnosis'
user-invocable: true
---

# FMoE Skill

## When To Use
- Run, review, or plan `fmoe_n4` experiments.
- Compare RouteRec / FMoE runs against `baseline_2` results.
- Interpret result files with `overall_seen_target` as the default evaluation view.
- Connect experiment findings to the current paper draft in `writing/ACM_template/sample-sigconf.tex`.
- Diagnose missing outputs, OOMs, stalled runs, or metric drift.
- Recommend the next focused experiment under the current paper-writing workflow.

## Session Start Context
Before making substantive claims, read these sources in this order:
1. `writing/ACM_template/sample-sigconf.tex`
2. `experiments/run/fmoe_n4/stageA_fmoe_n4.sh`
3. `experiments/run/fmoe_n4/stageB_fmoe_n4.sh`
4. `experiments/run/fmoe_n4/stageC_fmoe_n4.sh`
5. `experiments/run/fmoe_n4/stageD_fmoe_n4.sh`
6. `experiments/run/baseline_2/docs/baseline_2_best_valid_test_tables_v4.md`

If the user is asking about a specific run family, also inspect the corresponding logs or JSON results under:
- `experiments/run/artifacts/logs/fmoe_n4`
- `experiments/run/artifacts/results/fmoe_n4`
- `experiments/run/artifacts/logs/baseline_2`
- `experiments/run/artifacts/results/baseline_2`

## Current Working Assumptions
- Primary model family: `fmoe_n4`
- Primary baseline family: `baseline_2`
- Primary data root: `Datasets/processed/feature_added_v4`
- Primary evaluation view: `overall_seen_target`
- Secondary reporting views: `overall_unseen_target` and `overall`
- Primary paper artifact: `writing/ACM_template/sample-sigconf.tex`
- Primary goal: improve and finish the paper while keeping experiment interpretation consistent with the draft

## Operating Policy
- Keep `fmoe_n4` as the primary RouteRec track unless the user explicitly opens a new track.
- Keep `baseline_2` as the baseline comparison reference unless the user explicitly switches baseline families.
- Prioritize `seen target` metrics for model selection, claims, and iterative tuning.
- Keep `overall` and `unseen target` as supporting views rather than the default decision signal.
- Treat the paper draft as the source of truth for what results, figures, and analyses matter most.
- When the paper text and the current experimental protocol diverge, call out the mismatch explicitly.

## Data And Evaluation Protocol
- Default split: session-level chronological `70:15:15` train/validation/test.
- Held-out construction: keep train fixed, merge held-out sessions, sort by session start time, then split the held-out tail contiguously into validation and test.
- Current working rule: validation and test are effectively centered on `seen target` evaluation, with unseen targets dropped from the default decision path.
- If artifacts still include `overall` or `overall_unseen_target`, interpret them as supplementary reporting views and verify whether they reflect the latest preprocessing.
- Default metrics to summarize: HR, NDCG, and MRR at `k in {5, 10, 20}`.
- Default scalar for short summaries: `MRR@20` on `overall_seen_target`, unless the user requests another metric.

## Tuning Policy
- The common pattern is narrow tuning with most hyperparameters fixed.
- Search usually varies only LR, or at most 2 to 3 hyperparameters with a small discrete candidate set.
- Final training is treated as `100` epochs with `10` patience.
- Smaller budgets are for fast search or screening, not final paper claims.
- Hyperopt is currently common, but do not treat it as a permanent requirement.
- If the user is moving toward a bounded grid for paper clarity, help translate the current search space into explicit grid candidates instead of proposing open-ended search.

## Hydra Override Notes
- For standalone tune files without root seed, prefer `++seed=42`.
- For dotted search-space keys, use dict merge format:
  - `++search={rule_router.variant:[teacher_gls]}`

## Interpretation Priorities
- First ask which selection rule matters for the claim: best validation on `overall_seen_target`, best test on `overall_seen_target`, or a broader reporting table.
- For baseline comparisons, check whether the table uses best-validation selection or per-metric maxima before making statements.
- For RouteRec claims, tie any gain or regression back to the paper narrative: routing control, staged structure, cue ablation, transferability, or dataset-specific behavior.
- Flag paper drift when the draft still describes unseen-target retention or another protocol that does not match the current run setup.

## Recommended Next-Step Style
- Prefer one concrete run-family extension, one analysis/table update, and one paper-text update candidate.
- Avoid suggesting broad retuning unless the user asks for it.
- Prefer modifications that can be explained cleanly in the paper.

## Resources
- Paper draft: `writing/ACM_template/sample-sigconf.tex`
- FMoE run entrypoints: `experiments/run/fmoe_n4`
- Baseline summary tables: `experiments/run/baseline_2/docs/baseline_2_best_valid_test_tables_v4.md`
- FMoE logs/results: `experiments/run/artifacts/logs/fmoe_n4`, `experiments/run/artifacts/results/fmoe_n4`
- Baseline logs/results: `experiments/run/artifacts/logs/baseline_2`, `experiments/run/artifacts/results/baseline_2`
